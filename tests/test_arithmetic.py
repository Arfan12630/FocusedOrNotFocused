import json
import time

import numpy as np
import pytest
from PyQt6 import QtWidgets

from brainviz.arithmetic import MEASURED, PRACTICE, ArithmeticDialog, make_questions
from brainviz.recorder import read_csv
from brainviz.sim_source import SimSource
from brainviz.source import Channel, State, Status
from brainviz.ui import MainWindow


@pytest.fixture
def session(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(SimSource(seed=1), out_dir=tmp_path, skip_impedance=True)
    window.hub.configure([Channel(0, "Ch1"), Channel(1, "Ch2")], 250)
    window._setup_channels()
    window.status = Status(State.STREAMING)
    window.hub.rate.add(250, time.monotonic() - 1)
    window.hub.rate.add(250)
    dialog = ArithmeticDialog(window)
    window._task_dialog = dialog
    yield app, window, dialog
    dialog.finish_session("aborted")
    window.close()


def events(dialog):
    return [json.loads(line) for line in dialog.events_path.read_text().splitlines()]


def test_complete_session_pairs_eeg_with_behavior_and_practice_flags(session):
    _, window, dialog = session
    dialog.start_session()
    rec = dialog.rec
    assert dialog.active and not window.btn_record.isEnabled()
    window.hub.push(np.arange(5), np.zeros(5), np.ones((5, 2)))
    for i in range(PRACTICE + MEASURED):
        q = dialog.current
        dialog.answer(q["choices"].index(q["answer"]))
        dialog.answer(0)  # duplicated key/click during the gap cannot create another response
        dialog.next_question()
    assert not dialog.active and window.recorder is None
    records = events(dialog)
    responses = [e for e in records if e["event"] == "trial_response"]
    assert len(responses) == PRACTICE + MEASURED
    assert sum(e["phase"] == "practice" for e in responses) == PRACTICE
    assert all(e["correct"] and e["rt_s"] >= 0 for e in responses)
    assert all(a["host_monotonic_s"] <= b["host_monotonic_s"] for a, b in zip(records, records[1:]))
    meta = json.loads(rec.sidecar.read_text())
    assert meta["task"]["state"] == "complete"
    assert meta["task"]["events_file"] == dialog.events_path.name
    assert meta["simulated"] and meta["samples"] == 5
    assert read_csv(rec.path)[1].shape == (5, 6)


def test_timeout_and_abort_preserve_partial_session(session):
    _, window, dialog = session
    dialog.start_session()
    dialog.deadline = 0
    dialog.tick()
    response = [e for e in events(dialog) if e["event"] == "trial_response"][0]
    assert response["timed_out"] and response["response"] is None and response["rt_s"] is None
    dialog.next_question()
    dialog.reject()
    assert window.recorder is None
    assert events(dialog)[-2]["event"] == "trial_aborted"
    assert events(dialog)[-1]["state"] == "aborted"
    assert json.loads(dialog.rec.sidecar.read_text())["task"]["state"] == "aborted"


def test_disconnect_finalizes_events_even_if_main_window_stops_eeg_first(session):
    _, window, dialog = session
    dialog.start_session()
    window._on_status(Status(State.DISCONNECTED, "lost"))
    dialog.tick()
    assert not dialog.active
    assert events(dialog)[-1]["state"] == "interrupted"
    meta = json.loads(dialog.rec.sidecar.read_text())
    assert meta["task"]["state"] == "interrupted" and meta["samples"] == 0


def test_question_choices_are_unique_and_reproducible():
    assert make_questions(42) == make_questions(42)
    for q in make_questions(42):
        assert len(set(q["choices"])) == 4
        assert q["choices"].count(q["answer"]) == 1
