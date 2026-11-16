"""pyqtgraph UI: scrolling traces, band power, spectrogram, impedance check, recording controls.

Colour roles follow the dataviz method on a dark surface:
  * band bars     categorical slots 1-5 (validated), also direct-labelled on the axis
  * EEG traces    one neutral ink (identity comes from the labelled lanes, not hue)
  * spectrogram   single-hue sequential blue ramp
  * impedance     status palette, always paired with a glyph and a word
"""
from __future__ import annotations

import collections
import math
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QObject, Qt, pyqtSignal

from . import dsp
from .recorder import Recorder
from .ringbuffer import DataHub, RingBuffer
from .source import EEGSource, State, Status

# -- palette (dark mode) ---------------------------------------------------------------------
SURFACE, PAGE = "#1a1a19", "#0d0d0d"
INK, INK2, MUTED = "#f0efec", "#c3c2b7", "#898781"
GRID, AXIS = "#2c2c2a", "#383835"
BAND_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]  # blue, orange, aqua, yellow, magenta
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
# Sequential blue ramp, steps 700 -> 100 (dark surface: low values recede into the background).
SEQ_BLUE = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5",
            "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"]

# Provisional impedance thresholds (ohms). Tune once real Flex8 readings are seen.
IMPEDANCE_GOOD_OHM = 500e3
IMPEDANCE_FAIR_OHM = 2e6

SCALES_UV = [20, 50, 100, 200, 500]
WINDOWS_S = [5, 10, 20]
TICK_MS = 33
BAND_EVERY_S = 0.5
SPEC_EVERY_S = 0.25
SPEC_SECONDS = 60
SPEC_FMAX = 50.0
SPEC_FLOOR_DB = -20.0
BAND_WINDOW_S = 4.0

STATE_STYLE = {  # state -> (status colour role, glyph)
    State.IDLE: ("warning", "○"),
    State.SCANNING: ("warning", "◐"),
    State.CONNECTING: ("warning", "◐"),
    State.STREAMING: ("good", "●"),
    State.IMPEDANCE: ("warning", "◆"),
    State.DISCONNECTED: ("critical", "✕"),
    State.ERROR: ("critical", "✕"),
}


def classify_impedance(ohms: float) -> tuple[str, str, str]:
    """-> (status role, glyph, word). Never colour alone."""
    if not math.isfinite(ohms):
        return "critical", "—", "No signal"
    if ohms < IMPEDANCE_GOOD_OHM:
        return "good", "✓", "Good"
    if ohms < IMPEDANCE_FAIR_OHM:
        return "warning", "!", "Fair"
    return "critical", "✕", "Poor"


def format_ohms(ohms: float) -> str:
    if not math.isfinite(ohms):
        return "—"
    if ohms >= 10e6:
        return "> 10 MΩ"
    if ohms >= 1e6:
        return f"{ohms / 1e6:.2f} MΩ"
    return f"{ohms / 1e3:,.0f} kΩ"


def _swatch(color: str) -> QtGui.QIcon:
    pm = QtGui.QPixmap(14, 14)
    pm.fill(Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    p.setBrush(QtGui.QColor(color))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(1, 1, 12, 12)
    p.end()
    return QtGui.QIcon(pm)


def apply_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    for role, color in [
        (QtGui.QPalette.ColorRole.Window, PAGE), (QtGui.QPalette.ColorRole.Base, SURFACE),
        (QtGui.QPalette.ColorRole.AlternateBase, SURFACE), (QtGui.QPalette.ColorRole.Text, INK),
        (QtGui.QPalette.ColorRole.WindowText, INK), (QtGui.QPalette.ColorRole.ButtonText, INK),
        (QtGui.QPalette.ColorRole.Button, "#262625"), (QtGui.QPalette.ColorRole.ToolTipBase, SURFACE),
        (QtGui.QPalette.ColorRole.ToolTipText, INK), (QtGui.QPalette.ColorRole.Highlight, "#2a78d6"),
        (QtGui.QPalette.ColorRole.HighlightedText, "#ffffff"),
    ]:
        pal.setColor(role, QtGui.QColor(color))
    app.setPalette(pal)
    app.setStyleSheet(f"""
        QPushButton {{ padding: 5px 12px; border: 1px solid {AXIS}; border-radius: 4px; background: #262625; }}
        QPushButton:hover {{ background: #30302e; }}
        QPushButton:disabled {{ color: {MUTED}; background: {SURFACE}; }}
        QPushButton:checked {{ background: #3a1f1f; border-color: {STATUS['critical']}; }}
        QComboBox {{ padding: 3px 8px; border: 1px solid {AXIS}; border-radius: 4px; background: #262625; }}
        QStatusBar QLabel {{ padding: 0 8px; }}
        QTableWidget {{ background: {SURFACE}; gridline-color: {GRID}; border: 1px solid {AXIS}; }}
        QHeaderView::section {{ background: #262625; color: {INK2}; border: 0; padding: 4px; }}
    """)


def _style_plot(widget: pg.PlotWidget) -> pg.PlotItem:
    item = widget.getPlotItem()
    item.setMenuEnabled(False)
    item.hideButtons()
    item.setMouseEnabled(False, False)
    for name in ("left", "bottom"):
        ax = item.getAxis(name)
        ax.setPen(pg.mkPen(AXIS))
        ax.setTextPen(pg.mkPen(MUTED))
    return item


class Bridge(QObject):
    """Marshals source-thread callbacks onto the GUI thread (cross-thread emits are queued)."""
    status = pyqtSignal(object)
    impedance = pyqtSignal(object)
    ready = pyqtSignal()


class ImpedanceDialog(QtWidgets.QDialog):
    def __init__(self, labels: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Impedance check")
        self.setMinimumWidth(420)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        lay = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            "Press the headset electrodes against the scalp and wait for each channel to settle. "
            f"Aim for “Good” (below {IMPEDANCE_GOOD_OHM / 1e3:.0f} kΩ). The signal is paused while this window is open. "
            "Thresholds are provisional.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {INK2};")
        lay.addWidget(hint)
        self.table = QtWidgets.QTableWidget(len(labels), 3)
        self.table.setHorizontalHeaderLabels(["Electrode", "Impedance", "Status"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        for r, label in enumerate(labels):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(label))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem("waiting…"))
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(""))
        self.table.setFixedHeight(32 + 30 * len(labels))
        lay.addWidget(self.table)
        self.summary = QtWidgets.QLabel("Waiting for impedance data…")
        self.summary.setStyleSheet(f"color: {INK2};")
        lay.addWidget(self.summary)
        done = QtWidgets.QPushButton("Done, start signal")
        done.clicked.connect(self.accept)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(done)
        lay.addLayout(row)
        self.last_ohms: np.ndarray | None = None

    def update_values(self, ohms) -> None:
        ohms = np.asarray(ohms, dtype=float)
        self.last_ohms = ohms
        good = 0
        for r, v in enumerate(ohms[: self.table.rowCount()]):
            role, glyph, word = classify_impedance(float(v))
            good += role == "good"
            self.table.item(r, 1).setText(format_ohms(float(v)))
            item = self.table.item(r, 2)
            item.setText(f"{glyph} {word}")
            item.setIcon(_swatch(STATUS[role]))
        self.summary.setText(f"{good} of {len(ohms)} electrodes good")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, source: EEGSource, out_dir="recordings", notch: float | None = 60.0,
                 skip_impedance: bool = False) -> None:
        super().__init__()
        self.source = source
        self.out_dir = Path(out_dir)
        self.notch = notch or None
        self.skip_impedance = skip_impedance
        self.hub = DataHub(seconds=60.0)
        self.bridge = Bridge()
        self.recorder: Recorder | None = None
        self._task_dialog = None
        self.status = Status(State.IDLE)
        self._prev_state = State.IDLE
        self._impedance_offered = False
        self._imp_dialog: ImpedanceDialog | None = None
        self._last_impedance: np.ndarray | None = None
        self.tick_ms: collections.deque = collections.deque(maxlen=600)

        # per-channel display state (built in _setup_channels)
        self._ring_id = None
        self._labels: list[str] = []
        self._filter: dsp.StreamFilter | None = None
        self._disp: RingBuffer | None = None
        self._cursor = 0
        self._curves: list[pg.PlotDataItem] = []
        self._powers: dict | None = None
        self._spec = np.full((SPEC_SECONDS * 4, int(SPEC_FMAX * 2) + 1), SPEC_FLOOR_DB)
        self._spec_cols = 0
        self._spec_levels = (SPEC_FLOOR_DB, SPEC_FLOOR_DB + 40.0)
        self._last_band = self._last_spec = self._last_info = 0.0

        self.setWindowTitle("BrainBit Flex8" + ("  ·  SIMULATED DATA" if source.simulated else ""))
        self.resize(1400, 860)
        self._build_ui()

        source.on_data = self.hub.push
        source.on_ready = self._on_ready_thread
        source.on_status = self.bridge.status.emit
        source.on_impedance = self.bridge.impedance.emit
        self.bridge.ready.connect(self._setup_channels)
        self.bridge.status.connect(self._on_status)
        self.bridge.impedance.connect(self._on_impedance)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self._tick)

    # -- construction --------------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(10, 8, 10, 4)

        bar = QtWidgets.QHBoxLayout()
        self.btn_record = QtWidgets.QPushButton("● Record")
        self.btn_record.setCheckable(True)
        self.btn_record.setEnabled(False)
        self.btn_record.clicked.connect(self._toggle_record)
        self.btn_imp = QtWidgets.QPushButton("Impedance…")
        self.btn_imp.setEnabled(False)
        self.btn_imp.clicked.connect(self._open_impedance)
        self.btn_reconnect = QtWidgets.QPushButton("Reconnect")
        self.btn_reconnect.setEnabled(False)
        self.btn_reconnect.clicked.connect(self._reconnect)
        self.btn_task = QtWidgets.QPushButton("Arithmetic…")
        self.btn_task.clicked.connect(self.open_arithmetic)
        self.cmb_scale = QtWidgets.QComboBox()
        self.cmb_scale.addItems([f"±{s} µV" for s in SCALES_UV])
        self.cmb_scale.setCurrentIndex(2)
        self.cmb_scale.currentIndexChanged.connect(self._apply_view)
        self.cmb_window = QtWidgets.QComboBox()
        self.cmb_window.addItems([f"{w} s" for w in WINDOWS_S])
        self.cmb_window.setCurrentIndex(1)
        self.cmb_window.currentIndexChanged.connect(self._apply_view)
        self.cmb_channel = QtWidgets.QComboBox()
        self.cmb_channel.currentIndexChanged.connect(self._on_channel_changed)
        self.chk_abs = QtWidgets.QCheckBox("Absolute µV²")
        self.chk_abs.toggled.connect(self._refresh_bands)
        for w in (self.btn_record, self.btn_imp, self.btn_reconnect, self.btn_task):
            bar.addWidget(w)
        bar.addSpacing(18)
        for text, w in (("Scale", self.cmb_scale), ("Window", self.cmb_window), ("Channel", self.cmb_channel)):
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(f"color: {INK2};")
            bar.addWidget(lbl)
            bar.addWidget(w)
        bar.addWidget(self.chk_abs)
        bar.addStretch(1)
        outer.addLayout(bar)

        # traces
        self.trace_widget = pg.PlotWidget(background=SURFACE)
        self.trace_plot = _style_plot(self.trace_widget)
        self.trace_plot.showGrid(x=True, y=False, alpha=0.5)
        self.trace_plot.setLabel("bottom", "seconds", color=MUTED)
        self.trace_plot.setTitle("EEG · 1–45 Hz bandpass" + (f" + {self.notch:g} Hz notch" if self.notch else ""),
                                 color=INK2, size="10pt")

        # band power
        self.band_widget = pg.PlotWidget(background=SURFACE)
        self.band_plot = _style_plot(self.band_widget)
        self.bars = pg.BarGraphItem(x=np.arange(5), height=np.zeros(5), width=0.62,
                                    brushes=[pg.mkBrush(c) for c in BAND_COLORS], pen=pg.mkPen(None))
        self.band_plot.addItem(self.bars)
        self.band_plot.getAxis("bottom").setTicks([[(i, f"{n}\n{lo:g}–{hi:g} Hz")
                                                    for i, (n, (lo, hi)) in enumerate(dsp.BANDS.items())]])
        self.band_plot.getAxis("bottom").setHeight(46)  # room for the two-line band labels
        self.band_plot.setXRange(-0.6, 4.6, padding=0)
        # No gridlines: every bar carries its own value label, and pyqtgraph draws grids over bars.
        self.band_labels = []
        for i in range(5):
            t = pg.TextItem("", color=INK, anchor=(0.5, 1.0))
            self.band_plot.addItem(t)
            self.band_labels.append(t)

        # spectrogram
        self.spec_widget = pg.PlotWidget(background=SURFACE)
        self.spec_plot = _style_plot(self.spec_widget)
        self.spec_img = pg.ImageItem()
        self.spec_plot.addItem(self.spec_img)
        cmap = pg.ColorMap(np.linspace(0, 1, len(SEQ_BLUE)), [pg.mkColor(c).getRgb() for c in SEQ_BLUE])
        self.spec_img.setColorMap(cmap)
        # setRect scales by the image's current size, so the image must hold data BEFORE setRect.
        self.spec_img.setImage(self._spec, autoLevels=False, levels=self._spec_levels)
        self.spec_img.setRect(QtCore.QRectF(-SPEC_SECONDS, 0, SPEC_SECONDS, SPEC_FMAX))
        self.spec_plot.setXRange(-SPEC_SECONDS, 0, padding=0)
        self.spec_plot.setYRange(0, SPEC_FMAX, padding=0)
        self.spec_plot.setLabel("left", "Hz", color=MUTED)
        self.spec_plot.setLabel("bottom", "seconds", color=MUTED)
        self.colorbar = pg.ColorBarItem(values=self._spec_levels, colorMap=cmap, interactive=False,
                                        label="dB re 1 µV²/Hz", width=14)
        self.colorbar.setImageItem(self.spec_img, insert_in=self.spec_plot)
        try:
            self.colorbar.axis.setTextPen(pg.mkPen(MUTED))
            self.colorbar.axis.setPen(pg.mkPen(AXIS))
        except Exception:
            pass

        right = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.band_widget)
        right.addWidget(self.spec_widget)
        right.setSizes([380, 420])
        main = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(self.trace_widget)
        main.addWidget(right)
        main.setSizes([880, 480])
        outer.addWidget(main, 1)
        self.setCentralWidget(central)

        sb = self.statusBar()
        self.lbl_state = QtWidgets.QLabel("Idle")
        self.lbl_state.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_batt = QtWidgets.QLabel("")
        self.lbl_rate = QtWidgets.QLabel("")
        self.lbl_lost = QtWidgets.QLabel("")
        self.lbl_amp = QtWidgets.QLabel("")
        self.lbl_rec = QtWidgets.QLabel("")
        sb.addWidget(self.lbl_state, 1)
        for w in (self.lbl_amp, self.lbl_batt, self.lbl_rate, self.lbl_lost, self.lbl_rec):
            sb.addPermanentWidget(w)
        if self.source.simulated:
            sim = QtWidgets.QLabel(f"<span style='color:{STATUS['warning']}'>⚠</span> SIMULATED")
            sb.addPermanentWidget(sim)
        self._apply_view()

    # -- lifecycle -------------------------------------------------------------------------------
    def start(self) -> None:
        self.timer.start()
        self.source.start()

    def closeEvent(self, event) -> None:
        self.timer.stop()
        if self._task_dialog is not None:
            self._task_dialog.finish_session("aborted")
        if self.recorder is not None:
            self._stop_recording()
        self.source.stop()
        super().closeEvent(event)

    # -- source callbacks ---------------------------------------------------------------------------
    def _on_ready_thread(self, channels, fs) -> None:
        """Runs on the source thread, before the first sample."""
        self.hub.configure(channels, fs)
        self.bridge.ready.emit()

    def _setup_channels(self) -> None:
        ring = self.hub.ring
        if ring is None:
            return
        labels = [c.label for c in self.hub.channels]
        n, fs = len(labels), self.hub.fs
        if ring is self._ring_id and labels == self._labels:
            return
        self._ring_id, self._labels = ring, labels
        self._filter = dsp.StreamFilter(fs, n, notch=self.notch)
        self._disp = RingBuffer(n, ring.capacity)
        self._cursor = ring.total
        self._powers = None
        self.trace_plot.clear()
        self._curves = []
        for _ in range(n):
            c = pg.PlotDataItem(pen=pg.mkPen(INK2, width=1), connect="finite")
            c.setDownsampling(auto=True, method="peak")
            c.setClipToView(True)
            self.trace_plot.addItem(c)
            self._curves.append(c)
        self.cmb_channel.blockSignals(True)
        self.cmb_channel.clear()
        self.cmb_channel.addItem("All (mean)")
        self.cmb_channel.addItems(labels)
        self.cmb_channel.blockSignals(False)
        self._apply_view()
        self._on_channel_changed()

    def _on_status(self, st: Status) -> None:
        prev, self.status = self.status.state, st
        if st.state is State.STREAMING and prev is State.IMPEDANCE:
            self._reset_display()
        if st.state is State.STREAMING and prev is not State.STREAMING and not self._impedance_offered \
                and not self.skip_impedance:
            self._impedance_offered = True
            QtCore.QTimer.singleShot(500, self._open_impedance)
        if st.state is State.STREAMING and self._imp_dialog is not None \
                and st.message.startswith("Impedance check unavailable"):
            self._imp_dialog.reject()
        if st.state in (State.DISCONNECTED, State.ERROR) and self.recorder is not None:
            self._stop_recording(reason=f"recording closed: {st.state.value}")
        if st.state is State.IMPEDANCE:
            pass
        role, glyph = STATE_STYLE[st.state]
        msg = st.message.replace("<", "&lt;")
        self.lbl_state.setText(f"<span style='color:{STATUS[role]}'>{glyph}</span> "
                               f"<b>{st.state.value.capitalize()}</b>  {msg}")
        self.lbl_batt.setText(f"Battery {st.battery}%" if st.battery is not None else "")
        self.lbl_amp.setText(f"Amp: {st.amp_mode}" if st.amp_mode else "")
        self._update_buttons()

    def _on_impedance(self, ohms) -> None:
        self._last_impedance = np.asarray(ohms, dtype=float)
        if self._imp_dialog is not None:
            self._imp_dialog.update_values(self._last_impedance)

    def _update_buttons(self) -> None:
        streaming = self.status.state is State.STREAMING
        recording = self.recorder is not None
        task_active = self._task_dialog is not None and self._task_dialog.active
        self.btn_record.setEnabled((streaming or recording) and not task_active)
        self.btn_imp.setEnabled(streaming and not recording)
        self.btn_reconnect.setEnabled(self.status.state in (State.DISCONNECTED, State.ERROR, State.IDLE))
        self.btn_task.setEnabled(not recording or task_active)

    # -- actions -----------------------------------------------------------------------------------------
    def open_arithmetic(self) -> None:
        from .arithmetic import ArithmeticDialog
        if self._task_dialog is None:
            self._task_dialog = ArithmeticDialog(self)
            self._task_dialog.finished.connect(self._arithmetic_closed)
        self._task_dialog.show()
        self._task_dialog.raise_()
        self._task_dialog.activateWindow()

    def _arithmetic_closed(self, _result) -> None:
        self._task_dialog = None
        self._update_buttons()

    def _reconnect(self) -> None:
        self.source.reconnect()

    def _open_impedance(self) -> None:
        if self.status.state is not State.STREAMING or self.recorder is not None or self._imp_dialog is not None:
            return
        self._last_impedance = None
        dlg = ImpedanceDialog(self._labels, self)
        dlg.finished.connect(self._on_impedance_closed)
        self._imp_dialog = dlg
        self.source.start_impedance()
        dlg.open()

    def _on_impedance_closed(self, _result) -> None:
        self._imp_dialog = None
        self.source.stop_impedance()

    def _toggle_record(self, checked: bool) -> None:
        if checked:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self, task_meta: dict | None = None) -> None:
        if self.hub.ring is None:
            self.btn_record.setChecked(False)
            return
        meta = {
            "channels": [{"num": c.num, "label": c.label, "sdk_id": c.sdk_id, "sdk_name": c.sdk_name}
                         for c in self.hub.channels],
            "device": self.source.info,
            "display_filter": {"bandpass_hz": [1.0, 45.0], "notch_hz": self.notch,
                               "applied_to_csv": False},
            "last_impedance_ohms_at_start": None if self._last_impedance is None
            else [None if not math.isfinite(v) else float(v) for v in self._last_impedance],
        }
        if task_meta:
            meta["task"] = task_meta
        rec = Recorder(self.out_dir, self._labels, self.hub.fs, meta, simulated=self.source.simulated)
        path = rec.start()
        self.recorder = rec
        self.hub.recorder = rec
        self.btn_record.setText("■ Stop")
        self.btn_record.setChecked(True)
        self.statusBar().showMessage(f"Recording to {path}", 5000)
        self._update_buttons()

    def _stop_recording(self, reason: str = "") -> None:
        rec, self.recorder = self.recorder, None
        self.hub.recorder = None
        self.btn_record.setText("● Record")
        self.btn_record.setChecked(False)
        self.lbl_rec.setText("")
        if rec is not None:
            stats = self.hub.stats
            summary = rec.stop(extra={
                "lost_samples_estimate": stats.lost,
                "counter_resets": stats.resets,
                "measured_rate_hz": round(self.hub.rate.hz, 2),
                "last_impedance_ohms": None if self._last_impedance is None
                else [None if not math.isfinite(v) else float(v) for v in self._last_impedance],
            })
            note = f"Saved {summary.get('samples', 0)} samples to {rec.path}"
            if rec.error:
                note = f"Recording error: {rec.error}"
            self.statusBar().showMessage(f"{note} {reason}".strip(), 10000)
        self._update_buttons()

    # -- display ---------------------------------------------------------------------------------------
    @property
    def _scale(self) -> float:
        return float(SCALES_UV[self.cmb_scale.currentIndex()])

    @property
    def _window_s(self) -> float:
        return float(WINDOWS_S[self.cmb_window.currentIndex()])

    def _apply_view(self, *_args) -> None:
        n = len(self._labels) or 8
        scale = self._scale
        spacing = 2 * scale
        self.trace_plot.setXRange(-self._window_s, 0, padding=0)
        self.trace_plot.setYRange(-scale, (n - 1) * spacing + scale, padding=0)
        if self._labels:
            ticks = [((n - 1 - i) * spacing, label) for i, label in enumerate(self._labels)]
            self.trace_plot.getAxis("left").setTicks([ticks])

    def _reset_display(self) -> None:
        if self._filter is None or self._disp is None or self.hub.ring is None:
            return
        self._filter.reset()
        self._disp = RingBuffer(self._disp.n_ch, self._disp.capacity)
        self._cursor = self.hub.ring.total
        self._spec[:] = SPEC_FLOOR_DB
        self._spec_cols = 0
        self._powers = None
        for c in self._curves:
            c.setData([], [])

    def _selected(self) -> int:
        """-1 for the all-channel mean, else a channel index."""
        return self.cmb_channel.currentIndex() - 1

    def _on_channel_changed(self, *_args) -> None:
        sel = self._selected()
        name = "all channels (mean)" if sel < 0 else (self._labels[sel] if sel < len(self._labels) else "")
        self.band_plot.setTitle(f"Band power · {name}", color=INK2, size="10pt")
        self.spec_plot.setTitle(f"Spectrogram · {name}", color=INK2, size="10pt")
        for i, c in enumerate(self._curves):  # lift the selected lane, everything else stays neutral
            c.setPen(pg.mkPen(INK if i == sel else INK2, width=1.5 if i == sel else 1))
        self._spec[:] = SPEC_FLOOR_DB
        self._spec_cols = 0
        self._refresh_bands()

    def _refresh_bands(self, *_args) -> None:
        if self._powers is None:
            return
        sel = self._selected()
        absolute = np.array([self._powers[b].mean() if sel < 0 else self._powers[b][sel] for b in dsp.BANDS])
        total = absolute.sum()
        if self.chk_abs.isChecked():
            heights, fmt = absolute, "{:.1f}"
            self.band_plot.setLabel("left", "µV²", color=MUTED)
            self.band_plot.enableAutoRange("y", True)
        else:
            heights = 100.0 * absolute / total if total > 0 else np.zeros(5)
            fmt = "{:.0f}%"
            self.band_plot.setLabel("left", "% of 1–45 Hz power", color=MUTED)
            self.band_plot.setYRange(0, 100, padding=0.08)
        self.bars.setOpts(x=np.arange(5), height=heights, width=0.62)
        for i, (h, t) in enumerate(zip(heights, self.band_labels)):
            t.setText(fmt.format(h))
            t.setPos(i, h)

    def _tick(self) -> None:
        t_start = time.perf_counter()
        ring = self.hub.ring
        if ring is None or self._filter is None or self._disp is None or ring is not self._ring_id:
            return
        fs = self.hub.fs
        n_ch = self._disp.n_ch
        data, self._cursor, lost = ring.read_since(self._cursor)
        if lost:
            self._filter.reset()
            self._disp.push(np.full((min(lost, self._disp.capacity), n_ch), np.nan))
        if len(data) and self.status.state is State.STREAMING:
            self._disp.push(self._filter.process(data))

        self._draw_traces(fs)
        now = time.monotonic()
        if now - self._last_band >= BAND_EVERY_S:
            self._last_band = now
            self._update_bands(fs)
        if now - self._last_spec >= SPEC_EVERY_S:
            self._last_spec = now
            self._update_spectrogram(fs)
        if now - self._last_info >= 1.0:
            self._last_info = now
            self._update_info()
        self.tick_ms.append((time.perf_counter() - t_start) * 1000.0)

    def _draw_traces(self, fs: float) -> None:
        x_all = self._disp.latest(int(self._window_s * fs))
        n = len(x_all)
        if n == 0:
            return
        n_ch = self._disp.n_ch
        spacing = 2 * self._scale
        t = (np.arange(n) - n + 1) / fs
        for i, curve in enumerate(self._curves):
            curve.setData(t, x_all[:, i] + (n_ch - 1 - i) * spacing)

    def _update_bands(self, fs: float) -> None:
        x = self._disp.latest(int(BAND_WINDOW_S * fs))
        if len(x) < 2 * fs or not np.isfinite(x).all():
            return
        powers = dsp.band_powers(x, fs)
        if powers is not None:
            self._powers = powers
            self._refresh_bands()

    def _update_spectrogram(self, fs: float) -> None:
        x = self._disp.latest(int(2 * fs))
        if len(x) < int(2 * fs) or not np.isfinite(x).all():
            return
        sel = self._selected()
        cols = [x[:, sel]] if sel >= 0 else [x[:, i] for i in range(x.shape[1])]
        psds = [dsp.psd_column(c, fs, SPEC_FMAX)[1] for c in cols]
        db = 10.0 * np.log10(np.mean(psds, axis=0) + 1e-12)
        db = np.interp(np.arange(self._spec.shape[1]) * 0.5, dsp.psd_column(cols[0], fs, SPEC_FMAX)[0], db)
        self._spec = np.roll(self._spec, -1, axis=0)
        self._spec[-1] = db
        self._spec_cols = min(self._spec_cols + 1, self._spec.shape[0])
        filled = self._spec[-self._spec_cols:]
        lo, hi = np.percentile(filled, 3), np.percentile(filled, 99.5)
        hi = max(hi, lo + 20.0)
        a = 0.2  # smooth level changes so colours don't flicker
        self._spec_levels = (self._spec_levels[0] * (1 - a) + lo * a, self._spec_levels[1] * (1 - a) + hi * a)
        self.spec_img.setImage(self._spec, autoLevels=False)
        self.colorbar.setLevels(self._spec_levels)

    def _update_info(self) -> None:
        hz = self.hub.rate.hz
        self.lbl_rate.setText(f"{hz:.1f} Hz" if hz else "")
        lost = self.hub.stats.lost
        self.lbl_lost.setText(f"lost ≈{lost}" if lost else "")
        if self.recorder is not None:
            e = int(self.recorder.elapsed)
            self.lbl_rec.setText(f"<span style='color:{STATUS['critical']}'>●</span> REC {e // 60:02d}:{e % 60:02d}"
                                 f"  {self.recorder.samples_written} rows")
            if self.recorder.error:
                self._stop_recording(reason="(write error)")
