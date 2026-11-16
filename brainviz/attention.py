"""Qt gradCPT pilot: preflight, full-screen transitions, durable events and local summaries."""
from __future__ import annotations
import json
from pathlib import Path
import time

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .attention_analysis import process_session, write_json
from .attention_protocol import QualityConfig, PROTOCOL_VERSION, TRIAL_S, make_sequence
from .attention_store import DATA_ROOT, new_session_dir
from .quality import assess_quality
from .source import State

ASSETS = Path(__file__).parent / "assets" / "gradcpt"


class StimulusView(QtWidgets.QWidget):
    """Paint-time observations, explicitly not measured screen-photon timestamps."""
    def __init__(self, dialog):
        super().__init__(dialog)
        self.dialog = dialog
        self.images = {p.name: QtGui.QPixmap(str(p)) for p in ASSETS.glob("*.png")}
        if any(image.isNull() for image in self.images.values()) or len(self.images) != 21:
            raise RuntimeError("Official gradCPT images missing or unreadable")
        self.setMinimumSize(500, 400)
        self.last_trial = None
        self.last_paint = None
        self.last_opacity = 0.0
        self.feedback = ""

    def paintEvent(self, event):
        d = self.dialog
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#eeeeee"))
        if not d.active:
            painter.end()
            return
        now = time.perf_counter()
        index = max(0, min(int((now - d.started) / TRIAL_S), len(d.sequence) - 1))
        opacity = np.clip((now - d.started - index * TRIAL_S) / TRIAL_S, 0, 1)
        current = d.sequence[index]
        previous = d.sequence[index - 1]["image"] if index else "scrambled.png"
        size = min(450, self.width() - 100, self.height() - 150)
        rect = QtCore.QRect((self.width() - size) // 2, (self.height() - size) // 2, size, size)
        painter.drawPixmap(rect, self.images[previous])
        painter.setOpacity(float(opacity))
        painter.drawPixmap(rect, self.images[current["image"]])
        painter.setOpacity(1)
        painter.setPen(QtGui.QColor("#222222"))
        painter.drawText(20, 35, "PRACTICE — Space: city; withhold: mountain" if current["phase"].startswith("practice")
                         else "Space: city; withhold: mountain")
        if d.window.source.simulated:
            painter.drawText(20, 60, "SIMULATED EEG — NOT PARTICIPANT DATA")
        painter.drawText(20, self.height() - 35, self.feedback if current["phase"] == "practice" else "")
        painter.drawText(self.width() - 190, 35, "Esc: stop and save")
        painter.end()
        # Logging after QPainter submission provides an observable host paint boundary.
        submitted = time.perf_counter()
        d.presented(index, float(opacity), submitted, self.last_paint, self.last_trial)
        self.last_paint, self.last_trial, self.last_opacity = submitted, index, float(opacity)


class AttentionDialog(QtWidgets.QDialog):
    def __init__(self, window, root=DATA_ROOT, config=None, practice_trials=150, measured_trials=600):
        super().__init__(window)
        self.window = window
        self.root = Path(root)
        self.config = config or QualityConfig()
        self.practice_trials, self.measured_trials = practice_trials, measured_trials
        self.active = False
        self._finishing = False
        self.directory = None
        self.rec = None
        self.events = None
        self.summary = None
        self.meta = None
        self.sequence = []
        self.started = 0.0
        self.seen = set()
        self.onsets = {}
        self.presses = []
        self.last_press = -float("inf")
        self.quality = {"valid": False, "reasons": ["waiting_for_preview"]}
        self.preview_start = time.perf_counter()
        self.created_at = self.preview_start
        self.preview_total = window.hub.ring.total if window.hub.ring else 0
        self.preview_lost = window.hub.stats.lost
        self.preview_resets = window.hub.stats.resets
        self.setWindowTitle("Attention task — setup")
        self.resize(800, 700)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.setup = QtWidgets.QWidget()
        setup = QtWidgets.QVBoxLayout(self.setup)
        instructions = QtWidgets.QLabel(
            "Choose a recording channel, then 2 minutes practice and 8 minutes measured. "
            "Press Space for cities; do not press for mountains. Keep trying normally throughout. "
            "Labels describe relative response consistency, not verified focus. Data stays local.")
        instructions.setWordWrap(True)
        setup.addWidget(instructions)
        form = QtWidgets.QFormLayout()
        self.channel = QtWidgets.QComboBox()
        self.channel.addItem("All channels", None)
        self._channel_signature = None
        form.addRow("Record channel", self.channel)
        self.reference = QtWidgets.QLineEdit()
        self.ground = QtWidgets.QLineEdit()
        self.arrangement = QtWidgets.QLineEdit()
        self.arrangement.setPlaceholderText("Electrode locations, e.g. Ch2=F4; use unknown if not mapped")
        form.addRow("Channel / electrode arrangement", self.arrangement)
        form.addRow("Reference placement", self.reference)
        form.addRow("Ground placement", self.ground)
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setValue(time.time_ns() % (2**31))
        form.addRow("Sequence seed", self.seed)
        self.swing = QtWidgets.QDoubleSpinBox()
        self.swing.setRange(20, 10000)
        self.swing.setValue(self.config.swing_peak_to_peak_uv)
        form.addRow("Provisional swing threshold (µV p-p)", self.swing)
        setup.addLayout(form)
        self.check = QtWidgets.QPushButton("Check electrode impedance…")
        self.check.clicked.connect(window._open_impedance)
        setup.addWidget(self.check)
        self.impedance = QtWidgets.QLabel("Impedance: not checked (thresholds provisional)")
        self.impedance.setWordWrap(True)
        setup.addWidget(self.impedance)
        self.quality_label = QtWidgets.QLabel("Waiting for preview…")
        self.quality_label.setWordWrap(True)
        setup.addWidget(self.quality_label)
        retry = QtWidgets.QPushButton("Restart 10-second preview")
        retry.clicked.connect(self.reset_preview)
        setup.addWidget(retry)
        self.start_button = QtWidgets.QPushButton("Start practice and record")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_session)
        setup.addWidget(self.start_button)
        self.bypass_button = QtWidgets.QPushButton("Start without checks")
        self.bypass_button.setEnabled(False)
        self.bypass_button.clicked.connect(lambda: self.start_session(bypass=True))
        setup.addWidget(self.bypass_button)
        bypass_hint = QtWidgets.QLabel("Start without checks saves raw EEG immediately and marks the checks as skipped. "
                                      "Setup details can be left unknown. Quality flags are still calculated after the task.")
        bypass_hint.setWordWrap(True)
        setup.addWidget(bypass_hint)
        self.layout.addWidget(self.setup)
        self.view = StimulusView(self)
        self.view.hide()
        self.layout.addWidget(self.view)
        self.message = QtWidgets.QLabel("No recording started.")
        self.message.setWordWrap(True)
        self.layout.addWidget(self.message)
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        self.layout.addWidget(self.close_button)
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        self._last_preview_update = 0.0
        self._was_streaming = window.status.state is State.STREAMING
        self.channel.currentIndexChanged.connect(lambda _i: self.reset_preview())
        self.sync_channels()
        self.finished.connect(self._closed)
        try:
            setup_doc = json.loads((self.root / "setup.json").read_text())
            for name in ("reference", "ground", "arrangement"):
                getattr(self, name).setText(setup_doc.get(name, ""))
            self.swing.setValue(setup_doc.get("swing_peak_to_peak_uv", self.swing.value()))
        except (OSError, ValueError):
            pass

    def sync_channels(self):
        signature = [(c.num, c.label) for c in self.window.hub.channels]
        if signature == self._channel_signature:
            return
        wanted = self.channel.currentData()
        preferred = getattr(self.window, "attention_record_channel", None)
        self.channel.blockSignals(True)
        self.channel.clear()
        self.channel.addItem("All channels", None)
        for i, (num, label) in enumerate(signature):
            self.channel.addItem(f"{label} (channel {num + 1})", i)
        if preferred is not None and self._channel_signature in (None, []):
            wanted = next((i for i, (num, _label) in enumerate(signature) if num + 1 == preferred), None)
            if wanted is None:
                self.channel.addItem(f"Channel {preferred} (not available yet)", -1)
                wanted = -1
        if wanted is not None:
            index = self.channel.findData(wanted)
            if index >= 0:
                self.channel.setCurrentIndex(index)
        self.channel.blockSignals(False)
        self._channel_signature = signature

    def selected_indices(self):
        index = self.channel.currentData()
        if index is not None and not 0 <= index < len(self.window.hub.channels):
            return []
        return list(range(len(self.window.hub.channels))) if index is None else [index]

    def _closed(self, _result):
        self.timer.stop()
        self.finish_session("aborted")
        self.deleteLater()

    def reset_preview(self):
        self.preview_start = time.perf_counter()
        self.preview_total = self.window.hub.ring.total if self.window.hub.ring else 0
        self.preview_lost = self.window.hub.stats.lost
        self.preview_resets = self.window.hub.stats.resets
        self.quality = {"valid": False, "reasons": ["waiting_for_preview"]}
        self.start_button.setEnabled(False)

    def log(self, event, **fields):
        self.events.write(json.dumps({"event": event, "host_monotonic_s": time.perf_counter(),
                                     "host_time_unix": time.time(), **fields}, allow_nan=False) + "\n")
        self.events.flush()

    def tick(self):
        if self.active:
            try:
                if self.window.status.state is not State.STREAMING or self.window.recorder is not self.rec:
                    self.finish_session("interrupted")
                elif self.rec.error:
                    self.finish_session("error")
                elif time.perf_counter() - self.window.hub.last_receipt > self.config.stale_s:
                    self.finish_session("interrupted", "No recent EEG packets")
                elif time.perf_counter() - self.started >= len(self.sequence) * TRIAL_S:
                    self.finish_session("complete")
                else:
                    self.view.update()
            except (OSError, ValueError) as exc:
                self.finish_session("error", str(exc))
            return
        if self.directory is not None:
            return
        now = time.perf_counter()
        if now - self._last_preview_update < .5:
            return
        self._last_preview_update = now
        self.sync_channels()
        streaming = self.window.status.state is State.STREAMING
        if streaming and not self._was_streaming:
            self.reset_preview()
        self._was_streaming = streaming
        self.check.setEnabled(streaming and self.window.recorder is None)
        ohms = self.window._last_impedance
        fresh_impedance = ohms is not None and getattr(self.window, "_last_impedance_time", 0) >= self.created_at
        if fresh_impedance:
            self.impedance.setText("Impedance (provisional): " + ", ".join(
                f"{label}: {value / 1e3:.0f} kΩ" if np.isfinite(value) else f"{label}: no contact"
                for label, value in zip(self.window._labels, ohms)))
        ring, fs = self.window.hub.ring, self.window.hub.fs
        indices = self.selected_indices()
        can_record = (streaming and ring is not None and ring.total > 0 and bool(indices)
                      and now - self.window.hub.last_receipt <= self.config.stale_s
                      and self.window._imp_dialog is None and self.window.recorder is None)
        self.bypass_button.setEnabled(can_record)
        needed = round(self.config.preview_s * fs)
        count = min(needed, max(0, ring.total - self.preview_total)) if ring else 0
        elapsed = now - self.preview_start
        ready = streaming and count >= needed and elapsed >= self.config.preview_s
        if ready:
            from dataclasses import replace
            self.config = replace(self.config, swing_peak_to_peak_uv=self.swing.value())
            self.quality = assess_quality(ring.latest(needed)[:, indices], fs, gain=self.window.source.info.get("gain", 3),
                                          config=self.config, notch=self.window.notch,
                                          channel_labels=[self.window.hub.channels[i].label for i in indices])
            self.quality["source_channel_indices"] = indices
            if self.window.hub.stats.lost != self.preview_lost or self.window.hub.stats.resets != self.preview_resets:
                self.quality["valid"] = False
                self.quality["reasons"].append("missing_samples_or_reset_during_preview")
            if now - self.window.hub.last_receipt > self.config.stale_s:
                self.quality["valid"] = False
                self.quality["reasons"].append("stale_data")
            selection = ", ".join(self.window.hub.channels[i].label for i in indices)
            self.quality_label.setText(f"Preview ({selection}): " + ("no provisional artifact flags" if self.quality["valid"]
                                                      else ", ".join(self.quality["reasons"])))
        else:
            self.quality_label.setText(f"Preview: {min(count / fs, self.config.preview_s):.1f} / {self.config.preview_s:g} seconds of new EEG")
        placements = all(getattr(self, name).text().strip() for name in ("arrangement", "reference", "ground"))
        self.start_button.setEnabled(can_record and ready and self.quality["valid"] and bool(placements)
                                     and ohms is not None and self.window._imp_dialog is None
                                     and fresh_impedance and self.window.recorder is None)

    def start_session(self, _checked=False, bypass=False):
        self._last_preview_update = 0
        self.tick()  # Recheck freshness/settings at action time, not only on the last UI refresh.
        allowed = self.bypass_button.isEnabled() if bypass else self.start_button.isEnabled()
        if self.active or self.directory is not None or not allowed:
            return
        try:
            self.directory = new_session_dir(self.root, self.window.source.simulated)
            setup = {name: getattr(self, name).text().strip() or "unknown" for name in ("arrangement", "reference", "ground")}
            indices = self.selected_indices()
            setup["swing_peak_to_peak_uv"] = self.swing.value()
            write_json(self.root / "setup.json", setup)
            self.sequence = make_sequence(self.seed.value(), self.practice_trials, self.measured_trials)
            self.events = (self.directory / "events.jsonl").open("x", buffering=1)
            task_meta = {"name": "gradcpt", "protocol_version": PROTOCOL_VERSION, "seed": self.seed.value(),
                         "checks_skipped": bool(bypass), "source_channel_indices": indices}
            self.window._start_recording(task_meta, out_dir=self.directory, packet_timing=True,
                                         channel_indices=indices)
            self.rec = self.window.recorder
            if self.rec is None:
                raise RuntimeError("EEG recorder did not start")
            self.started = time.perf_counter() + .5  # allow full-screen layout before task begins
            for i, row in enumerate(self.sequence):
                row["scheduled_onset_s"] = self.started + i * TRIAL_S
            self.meta = {"session_id": self.directory.name, "state": "running", "task": task_meta,
                         "simulated": self.window.source.simulated, "fs_hz": self.window.hub.fs,
                         "gain": self.window.source.info.get("gain", 3), "notch_hz": self.window.notch,
                         "setup": setup, "quality_config": self.config.to_dict(), "preflight": self.quality,
                         "checks_skipped": bool(bypass), "source_channel_indices": indices,
                         "recorded_channels": [self.window.hub.channels[i].label for i in indices],
                         "impedance_ohms": None if self.window._last_impedance is None else
                         [float(v) if np.isfinite(v) else None for v in self.window._last_impedance],
                         "eeg_file": self.rec.path.name, "packet_timing_file": self.rec.packet_path.name,
                         "started_monotonic_s": self.started,
                         "alignment": "Monotonic host packet receipts and Qt paint submission times; approximate, "
                                      "not measured ADC sample times or screen presentation times.",
                         "stimuli": json.loads((ASSETS / "provenance.json").read_text())}
            write_json(self.directory / "sequence.json", self.sequence)
            write_json(self.directory / "session.json", self.meta)
            self.log("session_start", seed=self.seed.value())
            self.active = True
            self.setup.hide()
            self.message.hide()
            self.close_button.hide()
            self.view.show()
            self.showFullScreen()
            self.window._update_buttons()
            self.setFocus()
        except Exception as exc:
            self.finish_session("error", str(exc))
            self.message.setText(f"Session could not start: {exc}")

    def presented(self, index, opacity, now, previous_paint, previous_index):
        if not self.active or now < self.started:
            return
        current = self.sequence[index]
        try:
            if index not in self.seen:
                self.seen.add(index)
                self.onsets[index] = now
                if current["phase"].startswith("practice") and index >= 2:
                    from .gradcpt_scoring import score_trials
                    seen = [dict(t, onset_s=self.onsets[t["trial_id"]]) for t in self.sequence
                            if t["phase"].startswith("practice") and t["trial_id"] in self.seen]
                    finalized = [t for t in score_trials(seen, self.presses) if t["trial_id"] == index - 2]
                    if finalized:
                        row = finalized[0]
                        self.view.feedback = ("Correct: " if row["correct"] else "Missed: ") + (
                            "press Space for cities" if row["category"] == "city" else "withhold for mountains")
                self.log("trial_onset", trial_id=index, onset_s=now,
                         scheduled_onset_s=current["scheduled_onset_s"], phase=current["phase"])
                if now - current["scheduled_onset_s"] > self.config.frame_gap_s:
                    self.log("timing_failure", trial_ids=[index], reason="late_first_paint",
                             lateness_s=now - current["scheduled_onset_s"])
            if previous_paint is not None and now - previous_paint > self.config.frame_gap_s:
                self.log("timing_failure", trial_ids=list(range(max(0, previous_index or 0), index + 1)),
                         reason="paint_gap", gap_s=now - previous_paint)
            self.log("frame", trial_id=index, opacity=opacity, paint_time_s=now)
        except OSError as exc:
            self.finish_session("error", str(exc))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Escape and self.active:
            self.finish_session("aborted")
            return
        if event.key() == QtCore.Qt.Key.Key_Space and self.active:
            if event.isAutoRepeat() or self.view.last_trial is None:
                return
            now = time.perf_counter()
            if now < self.started or now - self.last_press < .05:
                return
            self.last_press = now
            row = self.sequence[self.view.last_trial]
            try:
                press = dict(time_s=now, trial_id=row["trial_id"], opacity=self.view.last_opacity, phase=row["phase"])
                self.log("press", **press)
                self.presses.append(press)
                if row["phase"] == "practice":
                    # Feedback is finalized using the same offline reference scorer.
                    from .gradcpt_scoring import score_trials
                    seen = [dict(t, onset_s=self.onsets[t["trial_id"]]) for t in self.sequence
                            if t["phase"].startswith("practice") and t["trial_id"] in self.seen]
                    scored = score_trials(seen, self.presses)
                    assigned = [r for r in scored if r["response_s"] == now]
                    self.view.feedback = "Correct" if assigned and assigned[-1]["correct"] else "Withhold for mountains"
            except OSError as exc:
                self.finish_session("error", str(exc))
            return
        super().keyPressEvent(event)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.ActivationChange and self.active and time.perf_counter() > self.started \
                and not self.isActiveWindow():
            self.finish_session("interrupted", "Task window lost focus")

    def finish_session(self, state, reason=""):
        if self._finishing or self.directory is None:
            return
        if self.meta is not None and self.meta.get("state") != "running":
            return
        self._finishing = True
        self.active = False
        self.timer.stop()
        errors = []
        ended = time.perf_counter()
        if self.events is not None:
            try:
                self.log("session_end", state=state, reason=reason)
            except OSError as exc:
                errors.append(str(exc))
                state = "error"
            finally:
                self.events.close()
                self.events = None
        if self.rec is not None:
            self.rec.meta["task"]["state"] = state
            try:
                if self.window.recorder is self.rec:
                    self.window._stop_recording(reason=f"Attention session {state}")
                else:
                    doc = json.loads(self.rec.sidecar.read_text())
                    self.rec._write_sidecar(doc | {"task": self.rec.meta["task"]})
                if self.rec.error:
                    errors.append(self.rec.error)
                    state = "error"
            except (OSError, RuntimeError) as exc:
                errors.append(str(exc))
                state = "error"
        if self.meta is not None:
            self.meta.update(state=state, ended_monotonic_s=ended, stop_reason=reason, errors=errors)
            try:
                write_json(self.directory / "session.json", self.meta)
                self.summary = process_session(self.directory)
            except Exception as exc:
                errors.append(str(exc))
                self.meta["processing_error"] = str(exc)
                try:
                    write_json(self.directory / "session.json", self.meta)
                except OSError:
                    pass
        self.view.hide()
        self.showNormal()
        self.message.show()
        self.close_button.show()
        self.close_button.setText("Close")
        s = self.summary
        if s:
            accuracy = "—" if s["accuracy"] is None else f"{s['accuracy']:.1%}"
            self.message.setText(f"{state.capitalize()}. Accuracy: {accuracy}. Mean correct city RT: "
                f"{s['mean_correct_city_rt_s'] or 0:.3f} s.\nTiming problem trials: {s['timing_problem_trials']}. "
                f"Usable EEG: {s['usable_eeg_duration_s']:.1f} s.\nWindow counts: {s['window_counts']}. "
                f"Labelling: {s['label_status']}.\nSaved locally: {self.directory}\n" + "; ".join(errors))
        else:
            self.message.setText(f"{state.capitalize()}: {reason}. Files kept in {self.directory}. " + "; ".join(errors))
        self.window._reset_display()
        self.window._update_buttons()
        self._finishing = False
