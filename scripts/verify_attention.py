"""Exercise the actual Qt task and recording pipeline with synthetic EEG/responses.

QT_QPA_PLATFORM=offscreen .venv/bin/python -P scripts/verify_attention.py [--full]
--full uses the production 150 practice + 600 measured trials (~10 minutes).
A simulator run never establishes headset signal quality or physical display timing.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

# Standard library takes precedence over unrelated root-level interview scripts.
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
from PyQt6 import QtCore, QtWidgets, QtTest
from brainviz.attention import AttentionDialog
from brainviz.attention_protocol import QualityConfig
from brainviz.attention_analysis import read_jsonl, write_json
from brainviz.attention_store import DATA_ROOT
from brainviz.sim_source import SimSource
from brainviz.source import State
from brainviz.ui import MainWindow, apply_theme

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--full', action='store_true')
parser.add_argument('--out', default=str(DATA_ROOT / 'verification'))
args = parser.parse_args()
app = QtWidgets.QApplication([])
apply_theme(app)
window = MainWindow(SimSource(seed=13), skip_impedance=True)
config = QualityConfig() if args.full else QualityConfig(preview_s=1, min_valid_responses=4)
dialog = AttentionDialog(window, root=args.out, config=config,
                         practice_trials=150 if args.full else 3, measured_trials=600 if args.full else 20)
window._attention_dialog = dialog
for name in ('reference', 'ground', 'arrangement'):
    getattr(dialog, name).setText('SIMULATED VERIFICATION — no physical electrodes')
window.show(); dialog.show(); window.start()
begin = time.perf_counter()
stage = 'connect'
responded = set()
last_report = 0
exit_code = 1


def drive():
    global stage, last_report, exit_code
    now = time.perf_counter()
    if now - begin > (680 if args.full else 60):
        dialog.finish_session('error', 'Verification timeout')
        print('FAIL: timeout', flush=True)
        window.close(); app.quit(); return
    if stage == 'connect' and window.status.state is State.STREAMING:
        window._open_impedance(); stage = 'impedance'
    elif stage == 'impedance' and window._imp_dialog is not None and window._last_impedance is not None:
        window._imp_dialog.accept(); stage = 'preflight'
    elif stage == 'preflight' and dialog.start_button.isEnabled():
        dialog.start_session(); stage = 'run'
        print('STARTED', dialog.directory, flush=True)
    elif stage == 'run' and dialog.active:
        i = dialog.view.last_trial
        if i is not None and i not in responded:
            trial = dialog.sequence[i]
            target = .56 + .12 * np.sin(i / 9)
            if now >= trial['scheduled_onset_s'] + target:
                responded.add(i)
                if trial['category'] == 'city':
                    QtTest.QTest.keyClick(dialog, QtCore.Qt.Key.Key_Space)
        if now - last_report >= 60:
            last_report = now
            print('RUNNING', f'{now-dialog.started:.1f}s', 'trial', i, flush=True)
    elif stage == 'run' and not dialog.active and dialog.summary is not None:
        frames = [e['paint_time_s'] for e in read_jsonl(dialog.directory / 'events.jsonl') if e['event'] == 'frame']
        gaps = np.diff(frames)
        result = {'full_protocol': args.full, 'platform': os.environ.get('QT_QPA_PLATFORM', 'native'),
                  'elapsed_s': now - begin, 'frame_gap_p95_s': float(np.percentile(gaps,95)),
                  'max_frame_gap_s': float(max(gaps)), 'summary': dialog.summary,
                  'limitation': 'Simulator/Qt submission timing only; not a headset pilot or photodiode display validation.'}
        write_json(dialog.directory / 'verification.json', result)
        print(json.dumps(result, indent=2), flush=True)
        exit_code = 0 if dialog.summary['state']=='complete' and dialog.summary['label_status']=='ok' else 1
        window.close(); app.quit()


timer=QtCore.QTimer();timer.setInterval(10);timer.timeout.connect(drive);timer.start()
app.exec()
raise SystemExit(exit_code)
