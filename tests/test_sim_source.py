import time

import numpy as np

from brainviz import dsp
from brainviz.sim_source import SIM_LABELS, EEGSynth, SimSource
from brainviz.source import State

from helpers import wait_until


def run_source(seconds, seed=1):
    src = SimSource(seed=seed)
    chunks, states, imps = [], [], []
    src.on_data = lambda p, m, uv: chunks.append((time.monotonic(), p, uv))
    src.on_status = lambda s: states.append(s.state)
    src.on_impedance = imps.append
    src.start()
    wait_until(lambda: State.STREAMING in states, 3, "STREAMING")
    time.sleep(seconds)
    return src, chunks, states, imps


def test_streams_at_the_nominal_rate_without_dropped_packet_numbers():
    src, chunks, states, _ = run_source(3.0)
    src.stop()
    total = sum(len(c[2]) for c in chunks)
    span = chunks[-1][0] - chunks[0][0]
    assert abs((total - len(chunks[0][2])) / span - 250) < 8
    packs = np.concatenate([c[1] for c in chunks])
    assert (np.diff(packs) == 1).all()
    assert states[:3] == [State.SCANNING, State.CONNECTING, State.STREAMING]


def test_data_arrives_as_microvolts_with_large_dc_offsets():
    src, chunks, *_ = run_source(1.0)
    src.stop()
    uv = np.vstack([c[2] for c in chunks])
    assert uv.shape[1] == len(SIM_LABELS)
    assert np.abs(uv.mean(axis=0)).max() > 1000  # tens of mV of DC, expressed in uV
    assert np.abs(uv.mean(axis=0)).max() < 100_000


def test_same_seed_gives_identical_signal_and_different_seed_does_not():
    a = EEGSynth(SIM_LABELS, 250.0, seed=5).generate(1000)
    b = EEGSynth(SIM_LABELS, 250.0, seed=5).generate(1000)
    c = EEGSynth(SIM_LABELS, 250.0, seed=6).generate(1000)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_generation_is_continuous_across_calls():
    whole = EEGSynth(SIM_LABELS, 250.0, seed=2).generate(1000)
    s = EEGSynth(SIM_LABELS, 250.0, seed=2)
    pieces = np.vstack([s.generate(n) for n in (1, 99, 400, 500)])
    # The random streams differ in draw order, so compare deterministic structure: same DC + drift + tones.
    assert pieces.shape == whole.shape
    assert abs(pieces.mean() - whole.mean()) < 1e-3
    assert np.abs(np.diff(pieces, axis=0)).max() < 1e-3  # no jumps at call boundaries


def test_occipital_alpha_and_frontal_blinks_and_mains_hum_are_present():
    synth = EEGSynth(SIM_LABELS, 250.0, seed=3)
    y = dsp.StreamFilter(250.0, 8, notch=None, band=(1, 100)).process(synth.generate(250 * 40) * 1e6)[250 * 5:]
    f, p = dsp.welch_psd(y, 250.0)
    o1, c3, fp1 = SIM_LABELS.index("O1"), SIM_LABELS.index("C3"), SIM_LABELS.index("Fp1")
    alpha = (f >= 8) & (f < 13)
    assert p[alpha, o1].sum() > 3 * p[alpha, c3].sum()
    hum = np.argmin(abs(f - 60))
    assert p[hum, c3] > 5 * np.median(p[(f > 50) & (f < 56), c3])
    assert np.abs(y[:, fp1]).max() > 60  # blink peaks, uV


def test_impedance_mode_pauses_data_reports_ohms_and_resumes():
    src, chunks, states, imps = run_source(0.5)
    src.start_impedance()
    wait_until(lambda: State.IMPEDANCE in states, 2, "IMPEDANCE")
    time.sleep(0.3)
    n = len(chunks)
    time.sleep(0.8)
    assert len(chunks) == n  # no samples while measuring
    assert len(imps) >= 3 and imps[-1].shape == (8,) and np.isfinite(imps[-1]).all()
    src.stop_impedance()
    wait_until(lambda: states[-1] is State.STREAMING, 2, "STREAMING again")
    time.sleep(0.4)
    assert len(chunks) > n
    src.stop()


def test_drop_and_reconnect():
    src, chunks, states, _ = run_source(0.3)
    src.drop()
    wait_until(lambda: states[-1] is State.DISCONNECTED, 2, "DISCONNECTED")
    n = len(chunks)
    time.sleep(0.3)
    assert len(chunks) == n
    src.reconnect()
    wait_until(lambda: states[-1] is State.STREAMING, 2, "STREAMING")
    time.sleep(0.3)
    assert len(chunks) > n
    src.stop()


def test_label_overrides_reach_the_channel_table():
    src = SimSource(seed=1)
    src.label_overrides = ["A", "B"]
    got = []
    src.on_ready = lambda ch, fs: got.append((ch, fs))
    src.start()
    wait_until(lambda: got, 3, "on_ready")
    src.stop()
    labels = [c.label for c in got[0][0]]
    assert labels[:2] == ["A", "B"] and labels[2:] == SIM_LABELS[2:] and got[0][1] == 250.0
