"""Offscreen end-to-end smoke test of the UI with the simulated source.

    QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/smoke_ui.py [outdir]

Drives the real MainWindow: auto impedance check, trace/band/spectrogram updates, recording, a dropped
connection and Reconnect. Prints tick-time statistics against the 33 ms budget and saves screenshots.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402
from PyQt6 import QtWidgets  # noqa: E402

from brainviz import dsp  # noqa: E402
from brainviz.recorder import read_csv  # noqa: E402
from brainviz.sim_source import SimSource  # noqa: E402
from brainviz.source import State  # noqa: E402
from brainviz.ui import MUTED, SURFACE, MainWindow, apply_theme  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else ".smoke")
out.mkdir(parents=True, exist_ok=True)

pg.setConfigOptions(antialias=False, background=SURFACE, foreground=MUTED)
app = QtWidgets.QApplication([])
apply_theme(app)
src = SimSource(seed=3)
win = MainWindow(src, out_dir=out / "rec", notch=60.0, skip_impedance=False)
win.show()
win.start()


def wait(seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.004)


def wait_for(cond, timeout: float, what: str) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if cond():
            return
        time.sleep(0.004)
    raise AssertionError(f"timed out waiting for {what} (state={win.status.state}, msg={win.status.message!r})")


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        raise AssertionError(msg)


print("1. connect + automatic impedance step")
wait_for(lambda: win.status.state is State.STREAMING, 5, "STREAMING")
wait_for(lambda: win._imp_dialog is not None, 3, "impedance dialog to open")
wait_for(lambda: win.status.state is State.IMPEDANCE, 3, "IMPEDANCE state")
wait(4.5)
dlg = win._imp_dialog
check(dlg is not None and dlg.last_ohms is not None, "dialog received impedance values")
dlg.grab().save(str(out / "impedance.png"))
n_before = win.hub.ring.total
dlg.accept()
wait_for(lambda: win.status.state is State.STREAMING, 3, "STREAMING after impedance")
check(win._imp_dialog is None, "dialog closed and cleared")
check(win._disp.latest(10000).shape[0] < 100, "display cleared after resuming signal")

print("2. traces, bands, spectrogram")
labels = [c.label for c in win.hub.channels]
win.cmb_channel.setCurrentIndex(1 + labels.index("O1"))
wait(6)
snapshots = []
for _ in range(14):  # sample the band powers over ~14 s (alpha amplitude is modulated slowly)
    wait(1.0)
    if win._powers is not None:
        snapshots.append(dsp.relative_powers(win._powers))
check(len(snapshots) >= 10, f"band powers computed ({len(snapshots)} snapshots)")
mean_rel = {k: np.mean([s[k] for s in snapshots], axis=0) for k in snapshots[0]}
io1, ic3 = labels.index("O1"), labels.index("C3")
check(mean_rel["alpha"][io1] > 2 * mean_rel["alpha"][ic3],
      f"alpha share O1 {mean_rel['alpha'][io1]:.0f}% > 2x C3 {mean_rel['alpha'][ic3]:.0f}%")
c3 = {k: float(v[ic3]) for k, v in mean_rel.items()}
check(max(c3, key=c3.get) == "delta", f"C3 background is 1/f, delta largest {({k: round(v) for k, v in c3.items()})}")
raw = win.hub.ring.latest(1500)
filt = win._disp.latest(1500)
f, p_raw = dsp.welch_psd(raw, 250.0)
_, p_filt = dsp.welch_psd(filt, 250.0)
i60 = int(np.argmin(abs(f - 60)))
drop_db = 10 * np.log10(p_raw[i60, 6] / p_filt[i60, 6])
check(drop_db > 40, f"60 Hz attenuated by {drop_db:.0f} dB")
check(abs(win.hub.rate.hz - 250) < 5, f"measured rate {win.hub.rate.hz:.1f} Hz")
check(win._spec_cols > 20, f"spectrogram has {win._spec_cols} columns")
rect = win.spec_img.mapRectToParent(win.spec_img.boundingRect())
check(abs(rect.left() + 60) < 0.5 and abs(rect.width() - 60) < 0.5 and abs(rect.height() - 50) < 0.5,
      f"spectrogram image maps to x -60..0 s, 0..50 Hz (got {rect.left():.1f}+{rect.width():.1f}, h={rect.height():.1f})")
qi = win.spec_img.qimage
colours = len({qi.pixel(x, y) for x in range(0, qi.width(), 3) for y in range(0, qi.height(), 3)})
check(colours > 50, f"spectrogram renders {colours} distinct colours")
win.grab().save(str(out / "main.png"))

print("3. recording")
win.btn_record.click()
check(win.recorder is not None, "recording started")
wait(3)
csv_path = win.recorder.path
win.btn_record.click()
check(win.recorder is None, "recording stopped")
header, data = read_csv(csv_path)
check(header[:4] == ["sample_index", "host_time_unix", "pack_num", "marker"] and header[4:] == labels, "CSV header")
check(abs(len(data) - 750) < 60, f"CSV rows {len(data)} ~ 750")
check(csv_path.with_suffix(".json").exists() and "-SIM" in csv_path.name, "sidecar written, filename flagged -SIM")

print("4. dropped connection and Reconnect")
win.btn_record.click()
wait(1)
src.drop()
wait_for(lambda: win.status.state is State.DISCONNECTED, 3, "DISCONNECTED")
check(win.recorder is None, "recording closed on disconnect")
check(win.btn_reconnect.isEnabled() and not win.btn_record.isEnabled(), "Reconnect enabled, Record disabled")
win.btn_reconnect.click()
wait_for(lambda: win.status.state is State.STREAMING, 3, "STREAMING after reconnect")
wait(1)

print("5. timing")
ticks = np.array(win.tick_ms)
over = float((ticks > 33).mean())
print(f"  tick ms: mean {ticks.mean():.2f}  p95 {np.percentile(ticks, 95):.2f}  max {ticks.max():.2f}  "
      f"over-budget {over:.1%}  (n={len(ticks)})")
check(np.percentile(ticks, 95) < 33, "p95 tick time inside the 33 ms budget")

win.close()
print("SMOKE OK, screenshots in", out.resolve())
