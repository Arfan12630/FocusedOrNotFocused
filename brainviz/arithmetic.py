"""Arithmetic pilot: behavioral events on the EEG recording's host clock."""
from __future__ import annotations

import json
import time

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .source import State

PRACTICE = 3
MEASURED = 20
TIMEOUT_S = 12.0


def make_questions(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    questions = []
    seen = set()
    while len(questions) < PRACTICE + MEASURED:
        a, b = int(rng.integers(20, 60)), int(rng.integers(2, 20))
        op = "+" if rng.integers(2) else "−"
        if (a, b, op) in seen:
            continue
        seen.add((a, b, op))
        answer = a + b if op == "+" else a - b
        offsets = rng.choice([-10, -5, -3, -2, -1, 1, 2, 3, 5, 10], 3, replace=False)
        choices = [answer, *(answer + int(d) for d in offsets)]
        rng.shuffle(choices)
        questions.append({"trial": len(questions) + 1,
                          "phase": "practice" if len(questions) < PRACTICE else "measured",
                          "a": a, "b": b, "operator": op, "answer": answer, "choices": choices})
    return questions


class ArithmeticDialog(QtWidgets.QDialog):
    def __init__(self, window) -> None:
        super().__init__(window)
        self.window = window
        self.active = False
        self.rec = None
        self.events = None
        self.events_path = None
        self.current = None
        self.index = -1
        self.results = []
        self.deadline = 0.0
        self.seed = time.time_ns() % (2**32)
        self.questions = make_questions(self.seed)
        self.setWindowTitle("Arithmetic + EEG" + (" — SIMULATED" if window.source.simulated else ""))
        self.resize(700, 470)
        layout = QtWidgets.QVBoxLayout(self)
        self.progress = QtWidgets.QLabel("3 practice questions, then 20 measured questions")
        layout.addWidget(self.progress)
        self.prompt = QtWidgets.QLabel("Solve each question mentally")
        self.prompt.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.prompt.setStyleSheet("font-size: 34px; padding: 24px;")
        layout.addWidget(self.prompt)
        self.hint = QtWidgets.QLabel(
            "Choose with keys 1–4 (or click). Answer as accurately and promptly as you can. "
            "Keep your head still and jaw relaxed. Each question allows 12 seconds. "
            "EEG and answers save automatically once you press Start. This is an arithmetic pilot; "
            "answers are not automatic focused/unfocused labels.")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        row = QtWidgets.QHBoxLayout()
        self.choices = []
        self.shortcuts = []
        for i in range(4):
            button = QtWidgets.QPushButton(f"{i + 1}: —")
            button.setMinimumHeight(65)
            button.setEnabled(False)
            button.clicked.connect(lambda _checked=False, j=i: self.answer(j))
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(str(i + 1)), self)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(lambda j=i: self.answer(j))
            self.shortcuts.append(shortcut)
            self.choices.append(button)
            row.addWidget(button)
        layout.addLayout(row)
        self.message = QtWidgets.QLabel("Waiting for live EEG…")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.start_button = QtWidgets.QPushButton("Start and record")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_session)
        layout.addWidget(self.start_button)
        self.stop_button = QtWidgets.QPushButton("Close")
        self.stop_button.clicked.connect(self.reject)
        layout.addWidget(self.stop_button)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(100)
        self.finished.connect(lambda _result: self.finish_session("aborted"))

    def log(self, event: str, **fields) -> None:
        # Host receipt timestamps permit approximate alignment, not hardware-trigger precision.
        record = {"event": event, "host_time_unix": time.time(),
                  "host_monotonic_s": time.perf_counter(), **fields}
        self.events.write(json.dumps(record) + "\n")
        self.events.flush()

    def start_session(self) -> None:
        if self.active or self.rec is not None or not self.can_start():
            return
        meta = {"name": "arithmetic_pilot", "seed": self.seed, "practice_trials": PRACTICE,
                "measured_trials": MEASURED, "timeout_s": TIMEOUT_S,
                "alignment": "Host event and EEG arrival Unix timestamps; approximate, no hardware triggers. "
                             "Onsets are GUI submission times, not measured screen presentation times.",
                "labels": "Behavior only; no validated attention labels generated.", "state": "running"}
        try:
            self.window._start_recording(task_meta=meta)
            self.rec = self.window.recorder
            if self.rec is None:
                raise RuntimeError("EEG recording could not start")
            self.events_path = self.rec.path.with_suffix(".events.jsonl")
            self.events = self.events_path.open("x", buffering=1)
            self.rec.meta["task"]["events_file"] = self.events_path.name
            self.rec._write_sidecar({"state": "recording"})
            self.active = True
            self.log("session_start", eeg_file=self.rec.path.name, seed=self.seed,
                     simulated=self.window.source.simulated)
            self.start_button.hide()
            self.stop_button.setText("Stop and save")
            self.window._update_buttons()
            self.next_question()
        except Exception as exc:
            self.finish_session("error")
            self.message.setText(f"Could not start recording: {exc}")

    def can_start(self) -> bool:
        return (self.window.status.state is State.STREAMING and self.window.hub.ring is not None
                and self.window.hub.rate.hz > 0 and self.window.recorder is None)

    def next_question(self) -> None:
        if not self.active:
            return
        self.index += 1
        if self.index == len(self.questions):
            self.finish_session("complete")
            return
        q = self.questions[self.index]
        self.progress.setText(f"Practice {self.index + 1}/{PRACTICE}" if q["phase"] == "practice"
                              else f"Measured question {self.index + 1 - PRACTICE}/{MEASURED}")
        self.prompt.setText(f"{q['a']} {q['operator']} {q['b']} = ?")
        for i, button in enumerate(self.choices):
            button.setText(f"{i + 1}: {q['choices'][i]}")
            button.setEnabled(True)
        self.message.setText("Select 1, 2, 3 or 4")
        self.current = q
        self.onset = time.perf_counter()
        self.deadline = self.onset + TIMEOUT_S
        self.log("trial_onset", **q)

    def answer(self, choice: int | None) -> None:
        if not self.active or self.current is None:
            return
        q, self.current = self.current, None
        response = None if choice is None else q["choices"][choice]
        result = {**q, "response": response, "choice_key": None if choice is None else choice + 1,
                  "correct": response == q["answer"], "timed_out": choice is None,
                  "rt_s": None if choice is None else time.perf_counter() - self.onset}
        self.results.append(result)
        self.log("trial_response", **result)
        for button in self.choices:
            button.setEnabled(False)
        if q["phase"] == "practice":
            self.message.setText("Correct" if result["correct"] else f"Answer: {q['answer']}")
        else:
            self.message.setText("Response saved" if choice is not None else "Time expired")
        self.prompt.setText("+")
        self.deadline = time.perf_counter() + 0.8

    def tick(self) -> None:
        if not self.active:
            if self.rec is None:
                ready = self.can_start()
                self.start_button.setEnabled(ready)
                self.message.setText("EEG connected. Ready to record." if ready else "Waiting for live EEG…")
            return
        if self.window.recorder is not self.rec or self.window.status.state is not State.STREAMING:
            self.finish_session("interrupted")
            return
        if self.rec.error:
            self.finish_session("error")
            return
        try:
            if time.perf_counter() >= self.deadline:
                if self.current is None:
                    self.next_question()
                else:
                    self.answer(None)
        except OSError as exc:
            self.finish_session("error")
            self.message.setText(f"Save error: {exc}")

    def finish_session(self, state: str) -> None:
        if self.rec is None:
            return
        if not self.active and self.rec.meta["task"]["state"] != "running":
            return
        self.active = False
        self.timer.stop()
        if self.events is not None:
            try:
                if self.current is not None:
                    self.log("trial_aborted", **self.current)
                self.log("session_end", state=state, answered_trials=len(self.results))
            except OSError:
                state = "error"
            finally:
                self.events.close()
                self.events = None
        self.current = None
        task = self.rec.meta["task"]
        task.update(state=state, answered_trials=len(self.results),
                    measured_answered=sum(q["phase"] == "measured" for q in self.results))
        if self.window.recorder is self.rec:
            self.window._stop_recording(reason=f"Arithmetic session {state}")
        else:
            # The main window may already have stopped the recording after a disconnect.
            doc = json.loads(self.rec.sidecar.read_text())
            self.rec._write_sidecar(doc | {"task": task})
        for button in self.choices:
            button.setEnabled(False)
        self.prompt.setText("Saved" if state == "complete" else "Session stopped")
        self.message.setText(f"{state.capitalize()}: {len(self.results)} responses.\nSaved to {self.rec.path}")
        self.stop_button.setText("Close")
        self.window._update_buttons()
