import numpy as np
import pytest

from brainviz import dsp

FS = 250.0


def tone(freq, seconds, amp=1.0, n_ch=1, dc=0.0):
    t = np.arange(int(seconds * FS)) / FS
    return np.tile((dc + amp * np.sin(2 * np.pi * freq * t))[:, None], (1, n_ch))


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def test_passes_10hz_and_rejects_60hz_and_dc():
    f = dsp.StreamFilter(FS, 1)
    y = f.process(tone(10, 20) + tone(60, 20, 5) + 20000.0)[int(5 * FS):]  # 20 mV DC offset
    # Re-measure each component through its own filter run so they don't mix.
    for freq, amp, lo, hi in [(10, 1.0, 0.89, 1.05), (60, 5.0, 0.0, 0.005)]:
        g = dsp.StreamFilter(FS, 1)
        out = g.process(tone(freq, 20, amp, dc=20000.0))[int(5 * FS):]
        gain = rms(out) / (amp / np.sqrt(2))
        assert lo <= gain <= hi, f"{freq} Hz gain {gain:.4f}"
    assert abs(y.mean()) < 0.5  # DC gone


def test_notch_can_be_disabled_and_moved():
    f50 = dsp.StreamFilter(FS, 1, notch=50.0)
    out = f50.process(tone(50, 20, 5))[int(5 * FS):]
    assert rms(out) / (5 / np.sqrt(2)) < 0.005
    no_notch = dsp.StreamFilter(FS, 1, notch=None, band=(1, 100))
    assert rms(no_notch.process(tone(60, 20, 5))[int(5 * FS):]) / (5 / np.sqrt(2)) > 0.9


def test_block_processing_equals_one_shot():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2500, 3)) * 10 + 5000
    whole = dsp.StreamFilter(FS, 3).process(x)
    parts = dsp.StreamFilter(FS, 3)
    pieces = [parts.process(x[i:i + n]) for i, n in [(0, 7), (7, 250), (257, 1), (258, 1000), (1258, 1242)]]
    np.testing.assert_allclose(np.vstack(pieces), whole, atol=1e-9)


def test_seeding_removes_startup_transient_from_dc():
    f = dsp.StreamFilter(FS, 2)
    y = f.process(np.full((500, 2), 30000.0))  # 30 mV in microvolts, perfectly constant
    assert np.abs(y).max() < 1e-6


def test_reset_reseeds():
    f = dsp.StreamFilter(FS, 1)
    f.process(np.full((100, 1), 1000.0))
    f.reset()
    y = f.process(np.full((100, 1), -4000.0))
    assert np.abs(y).max() < 1e-6


def test_channels_are_independent():
    f = dsp.StreamFilter(FS, 2)
    x = np.stack([tone(10, 10)[:, 0], np.zeros(int(10 * FS))], axis=1)
    y = f.process(x)
    assert np.abs(y[:, 1]).max() < 1e-9 and rms(y[:, 0]) > 0.5


def test_bad_arguments():
    with pytest.raises(ValueError):
        dsp.StreamFilter(FS, 1, band=(1, 200))
    with pytest.raises(ValueError):
        dsp.StreamFilter(FS, 1, notch=130)
    with pytest.raises(ValueError):
        dsp.StreamFilter(FS, 2).process(np.zeros((10, 3)))


def test_band_power_known_answer():
    amp = 10.0  # uV; a sine has power amp^2 / 2 = 50 uV^2
    p = dsp.band_powers(tone(10, 8, amp, n_ch=2), FS)
    assert p["alpha"][0] == pytest.approx(50.0, rel=0.05)
    for other in ("delta", "theta", "beta", "gamma"):
        assert p[other][0] < 0.5


def test_band_power_assigns_each_tone_to_its_band():
    for freq, band in [(2, "delta"), (6, "theta"), (10, "alpha"), (20, "beta"), (38, "gamma")]:
        p = dsp.band_powers(tone(freq, 8, 10), FS)
        assert max(p, key=lambda b: p[b][0]) == band, freq


def test_band_power_needs_enough_data_and_accepts_1d():
    assert dsp.band_powers(np.zeros(100), FS) is None
    assert dsp.band_powers(tone(10, 4)[:, 0], FS) is not None


def test_relative_powers_sum_to_100():
    rng = np.random.default_rng(1)
    rel = dsp.relative_powers(dsp.band_powers(rng.standard_normal((1500, 3)), FS))
    np.testing.assert_allclose(sum(rel.values()), 100.0)


def test_psd_column_peaks_at_tone_with_correct_level():
    x = tone(10, 2, 10)[:, 0]
    f, p = dsp.psd_column(x, FS, fmax=50)
    assert f[-1] <= 50 and abs(f[np.argmax(p)] - 10) <= 0.5
    assert np.sum(p) * (f[1] - f[0]) == pytest.approx(50.0, rel=0.1)  # total power of the sine
