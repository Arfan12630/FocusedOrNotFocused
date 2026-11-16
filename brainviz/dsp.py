"""Filtering and spectral analysis. Pure NumPy/SciPy, no Qt."""
from __future__ import annotations

import numpy as np
from scipy import signal

BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


class StreamFilter:
    """Streaming 1-45 Hz Butterworth bandpass plus a mains notch, with per-channel state.

    process() takes consecutive (n, n_ch) blocks and returns filtered blocks of the same shape.
    The first block seeds the filter state from its first sample, which removes the start-up
    transient that a large DC offset would otherwise cause. Call reset() after any gap in the data.
    """

    def __init__(self, fs: float, n_ch: int, band: tuple[float, float] = (1.0, 45.0),
                 notch: float | None = 60.0, order: int = 4, notch_q: float = 30.0) -> None:
        nyq = fs / 2.0
        if not 0 < band[0] < band[1] < nyq:
            raise ValueError(f"band {band} must lie inside (0, {nyq}) Hz")
        sos = signal.butter(order, band, btype="band", fs=fs, output="sos")
        if notch:
            if not 0 < notch < nyq:
                raise ValueError(f"notch {notch} Hz must lie inside (0, {nyq}) Hz")
            b, a = signal.iirnotch(notch, notch_q, fs=fs)
            sos = np.vstack([sos, signal.tf2sos(b, a)])
        self.fs, self.n_ch, self.band, self.notch = fs, n_ch, band, notch
        self.sos = sos
        self._zi_unit = signal.sosfilt_zi(sos)  # (sections, 2): steady state for a unit step
        self._zi: np.ndarray | None = None

    def reset(self) -> None:
        self._zi = None

    def process(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_ch:
            raise ValueError(f"expected (n, {self.n_ch}) samples, got {x.shape}")
        if len(x) == 0:
            return x
        if self._zi is None:
            # sosfilt with axis=0 wants state shaped (sections, 2, n_ch)
            self._zi = self._zi_unit[:, :, None] * x[0][None, None, :]
        y, self._zi = signal.sosfilt(self.sos, x, axis=0, zi=self._zi)
        return y


def welch_psd(x, fs: float, seg_s: float = 2.0):
    """Welch PSD in (unit^2)/Hz along axis 0. Returns (freqs, psd) or (None, None) if < 1 s of data."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < fs:
        return None, None
    nper = min(len(x), int(round(fs * seg_s)))
    return signal.welch(x, fs=fs, window="hann", nperseg=nper, noverlap=nper // 2,
                        detrend="constant", axis=0)


def band_powers(x, fs: float, bands: dict[str, tuple[float, float]] = BANDS):
    """Absolute band power (unit^2) per band, per channel. x is (n,) or (n, n_ch).

    Returns {band: array(n_ch)} or None when there is not enough data.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    f, p = welch_psd(x, fs)
    if f is None:
        return None
    df = f[1] - f[0]
    return {name: p[(f >= lo) & (f < hi)].sum(axis=0) * df for name, (lo, hi) in bands.items()}


def relative_powers(powers: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Each band as a percentage of the total over all bands."""
    total = sum(powers.values())
    total = np.where(total > 0, total, np.nan)
    return {name: 100.0 * v / total for name, v in powers.items()}


def psd_column(x, fs: float, fmax: float = 50.0):
    """One spectrogram column: Hann-windowed periodogram of a 1-D segment, density-scaled.

    Returns (freqs <= fmax, psd).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    w = np.hanning(n)
    spec = np.fft.rfft((x - x.mean()) * w)
    psd = (np.abs(spec) ** 2) / (fs * np.sum(w ** 2))
    psd[1:-1] *= 2.0  # one-sided
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    keep = freqs <= fmax
    return freqs[keep], psd[keep]
