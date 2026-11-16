"""Simulated Flex8: fake EEG that travels the same path as real data (volts in SDK-shaped packets)."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
from scipy import signal

from .source import Channel, EEGSource, State, packets_to_array

SIM_LABELS = ["Fp1", "Fp2", "C3", "C4", "T3", "T4", "O1", "O2"]


class EEGSynth:
    """Deterministic (seeded), stateful EEG generator. generate(n) returns volts, shape (n, n_ch)."""

    def __init__(self, labels=SIM_LABELS, fs: float = 250.0, seed: int | None = None) -> None:
        self.labels = list(labels)
        self.fs = fs
        self.rng = np.random.default_rng(seed)
        n = len(self.labels)
        rng = self.rng
        self.k = 0  # absolute sample counter

        def starts(prefix):
            return np.array([l.upper().startswith(prefix) for l in self.labels])

        occipital, parietal, frontal_pole, frontal = starts("O"), starts("P"), starts("FP"), starts("F")
        self.alpha_amp = np.where(occipital, 14e-6, np.where(parietal, 8e-6, 3e-6))
        self.alpha_f = 10.0 + rng.uniform(-0.3, 0.3, n)
        self.alpha_phi = rng.uniform(0, 2 * np.pi, n)
        self.alpha_mod_f = rng.uniform(0.05, 0.15, n)
        self.alpha_mod_phi = rng.uniform(0, 2 * np.pi, n)
        self.beta_f = 20.0 + rng.uniform(-3, 3, n)
        self.beta_phi = rng.uniform(0, 2 * np.pi, n)
        self.beta_amp = 1.5e-6
        # 1/f background: equal-variance AR(1) processes with log-spaced corner frequencies sum to a
        # 1/f spectrum (a Lorentzian per corner; the geometric spacing makes the sum fall as 1/f).
        self.noise_sigma = 2.5e-6  # per component; total background is ~sqrt(8) x this
        corners = 0.3 * 2.0 ** np.arange(8)  # 0.3 ... 38.4 Hz
        self._ar_a = np.exp(-2 * np.pi * corners / fs)
        self._ar_zi = [np.zeros((1, n)) for _ in corners]
        # Dry-electrode style DC offset (tens of mV) and slow drift; both must be filtered out.
        self.dc = rng.uniform(-30e-3, 30e-3, n)
        self.drift_amp = rng.uniform(0.5e-3, 2e-3, n)
        self.drift_f = rng.uniform(0.03, 0.08, n)
        self.drift_phi = rng.uniform(0, 2 * np.pi, n)
        self.hum_f, self.hum_amp = 60.0, 8e-6
        self.hum_phi = rng.uniform(0, 2 * np.pi, n)
        self.blink_weight = np.where(frontal_pole, 1.0, np.where(frontal, 0.5, 0.12))
        self.blink_amp = 120e-6
        self._blinks: list[float] = []
        self._next_blink = rng.uniform(1.0, 3.0)

    def generate(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty((0, len(self.labels)))
        fs, rng = self.fs, self.rng
        t = ((self.k + np.arange(n)) / fs)[:, None]
        two_pi = 2 * np.pi

        noise = np.zeros((n, len(self.labels)))
        for k, a in enumerate(self._ar_a):
            y, self._ar_zi[k] = signal.lfilter([np.sqrt(1 - a * a)], [1.0, -a],
                                               rng.standard_normal((n, len(self.labels))),
                                               axis=0, zi=self._ar_zi[k])
            noise += y
        env = 0.5 + 0.5 * np.sin(two_pi * self.alpha_mod_f * t + self.alpha_mod_phi)
        alpha = self.alpha_amp * env * np.sin(two_pi * self.alpha_f * t + self.alpha_phi)
        beta = self.beta_amp * (0.6 + 0.4 * np.sin(two_pi * 0.2 * t)) * np.sin(two_pi * self.beta_f * t + self.beta_phi)
        hum = self.hum_amp * np.sin(two_pi * self.hum_f * t + self.hum_phi)
        drift = self.drift_amp * np.sin(two_pi * self.drift_f * t + self.drift_phi)

        t_end = float(t[-1, 0])
        while self._next_blink < t_end + 1.0:
            self._blinks.append(self._next_blink)
            self._next_blink += rng.uniform(3.0, 6.0)
        self._blinks = [b for b in self._blinks if b > float(t[0, 0]) - 1.0]
        blink = np.zeros((n, len(self.labels)))
        for b in self._blinks:
            blink += np.exp(-0.5 * ((t - b) / 0.06) ** 2) * self.blink_amp * self.blink_weight

        self.k += n
        return self.dc + drift + self.noise_sigma * noise + alpha + beta + hum + blink


class ImpedanceSim:
    """Per-channel impedance that starts high and settles, one channel stays poor."""

    def __init__(self, n_ch: int, seed: int | None = None, tau: float = 4.0) -> None:
        rng = np.random.default_rng(None if seed is None else seed + 1)
        self.target = rng.uniform(60e3, 900e3, n_ch)
        self.target[min(5, n_ch - 1)] = 2.8e6
        self.start = 6e6
        self.tau = tau
        self.rng = rng

    def ohms(self, t: float) -> np.ndarray:
        base = self.target + (self.start - self.target) * np.exp(-t / self.tau)
        return base * (1 + 0.03 * self.rng.standard_normal(len(base)))


class SimSource(EEGSource):
    simulated = True
    PACKET_PERIOD_S = 0.04  # ~10 samples per callback at 250 Hz

    def __init__(self, seed: int | None = None, fs: float = 250.0, labels=SIM_LABELS) -> None:
        super().__init__()
        self.fs = fs
        self.seed = seed
        self._labels = list(labels)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._want = "signal"  # signal | impedance | dropped
        self.info = {"device_name": "Simulated Flex8", "simulated": True, "seed": seed}

    # -- public API -----------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="sim-source", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    def reconnect(self) -> None:
        if self._want == "dropped":
            self._want = "signal"

    def start_impedance(self) -> None:
        if self._want == "signal":
            self._want = "impedance"

    def stop_impedance(self) -> None:
        if self._want == "impedance":
            self._want = "signal"

    def drop(self) -> None:
        """Simulate losing the connection (for testing the Reconnect path)."""
        self._want = "dropped"

    # -- generator thread --------------------------------------------------------
    def _run(self) -> None:
        self._emit_status(State.SCANNING, "Simulated device")
        if self._stop.wait(0.3):
            return
        self._emit_status(State.CONNECTING, "Connecting (simulated)")
        if self._stop.wait(0.3):
            return
        channels = [Channel(num=i, label=l) for i, l in enumerate(self._labels)]
        synth = EEGSynth(self._labels, self.fs, self.seed)
        imp = ImpedanceSim(len(channels), self.seed)
        nums = [c.num for c in channels]
        self._publish_ready(channels, self.fs)
        self._emit_status(State.STREAMING, f"Simulated EEG, {len(channels)} channels @ {self.fs:g} Hz",
                          battery=87, amp_mode="Signal")

        current, pack, emitted = "signal", 0, 0
        t_ref = imp_t0 = last_imp = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if self._want != current:
                current = self._want
                if current == "impedance":
                    imp_t0, last_imp = now, 0.0
                    self._emit_status(State.IMPEDANCE, "Measuring impedance (simulated, signal paused)",
                                      amp_mode="Resist")
                elif current == "signal":
                    t_ref, emitted = now, 0
                    self._emit_status(State.STREAMING, "Signal resumed (simulated)", amp_mode="Signal")
                else:
                    self._emit_status(State.DISCONNECTED, "Simulated connection lost. Press Reconnect.")

            if current == "signal":
                n = min(int((now - t_ref) * self.fs) - emitted, int(self.fs * 2))
                if n > 0:
                    volts = synth.generate(n)
                    packets = [SimpleNamespace(PackNum=(pack + i) & 0xFFFFFF, Marker=0, Samples=row.tolist())
                               for i, row in enumerate(volts)]
                    pack += n
                    emitted += n
                    p, m, uv = packets_to_array(packets, nums)
                    if self.on_data:
                        self.on_data(p, m, uv)
            elif current == "impedance" and now - last_imp >= 0.2:
                last_imp = now
                if self.on_impedance:
                    self.on_impedance(imp.ohms(now - imp_t0))
            self._stop.wait(self.PACKET_PERIOD_S)
