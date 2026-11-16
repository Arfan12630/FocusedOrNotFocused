import json

import numpy as np

from brainviz.recorder import Recorder, read_csv


def make(tmp_path, simulated=False, labels=("Fp1", "O1")):
    return Recorder(tmp_path, list(labels), 250.0, meta={"device": {"serial": "X1"}}, simulated=simulated)


def push_rows(rec, start, n, host=1000.0):
    idx = np.arange(start, start + n)
    uv = np.stack([idx * 1.0, -idx * 0.5], axis=1)
    rec.push(idx + 100, np.zeros(n, int), uv, host)


def test_round_trip_header_values_and_sidecar(tmp_path):
    rec = make(tmp_path)
    path = rec.start()
    push_rows(rec, 0, 5)
    push_rows(rec, 5, 3, host=1000.04)
    summary = rec.stop(extra={"lost_samples_estimate": 0})
    header, data = read_csv(path)
    assert header == ["sample_index", "host_time_unix", "pack_num", "marker", "Fp1", "O1"]
    assert data.shape == (8, 6)
    np.testing.assert_array_equal(data[:, 0], np.arange(8))
    np.testing.assert_array_equal(data[:, 2], np.arange(8) + 100)
    np.testing.assert_allclose(data[:, 4], np.arange(8.0))
    np.testing.assert_allclose(data[:, 5], -np.arange(8) * 0.5)
    assert data[0, 1] == 1000.0 and data[7, 1] == 1000.04  # samples in one packet share the arrival time
    assert summary["samples"] == 8

    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["state"] == "complete" and meta["samples"] == 8
    assert meta["simulated"] is False and meta["fs_hz"] == 250.0
    assert meta["device"] == {"serial": "X1"} and meta["lost_samples_estimate"] == 0
    assert meta["columns"] == header and "unfiltered" in meta["note"]


def test_simulated_recordings_are_flagged_in_name_and_sidecar(tmp_path):
    rec = make(tmp_path, simulated=True)
    path = rec.start()
    rec.stop()
    assert path.name.endswith("-SIM.csv")
    assert json.loads(path.with_suffix(".json").read_text())["simulated"] is True


def test_same_second_start_does_not_overwrite(tmp_path):
    a, b = make(tmp_path), make(tmp_path)
    pa, pb = a.start(), b.start()
    a.stop(), b.stop()
    assert pa != pb and pa.exists() and pb.exists()


def test_pushes_before_start_and_after_stop_are_ignored(tmp_path):
    rec = make(tmp_path)
    push_rows(rec, 0, 3)  # not started
    path = rec.start()
    push_rows(rec, 0, 2)
    rec.stop()
    push_rows(rec, 0, 4)  # after stop
    assert read_csv(path)[1].shape[0] == 2


def test_labels_with_commas_are_quoted(tmp_path):
    rec = make(tmp_path, labels=("a,b", "c"))
    path = rec.start()
    push_rows(rec, 0, 1)
    rec.stop()
    assert path.read_text().splitlines()[0].endswith('"a,b",c')


def test_write_failure_is_reported_not_raised(tmp_path):
    rec = make(tmp_path)
    path = rec.start()
    rec._q.put(("bad", "bad", "bad", "bad"))  # the writer cannot format this
    summary = rec.stop()
    assert rec.error is not None
    assert json.loads(path.with_suffix(".json").read_text())["state"] == "error"
    assert "samples" in summary
