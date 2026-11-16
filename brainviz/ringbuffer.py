"""Thread-safe sample ring buffer, plus the DataHub that sits between a source and the UI/recorder."""
from __future__ import annotations

import collections
import threading
import time

import numpy as np


class RingBuffer:
    """Fixed-capacity (samples x channels) buffer with a monotonic absolute sample counter.

    Writers call push(); readers keep a cursor (an absolute sample index) and call read_since().
    If a reader falls more than `capacity` samples behind, the oldest data is gone: read_since
    returns what is left and reports how many samples were lost.
    """

    def __init__(self, n_ch: int, capacity: int) -> None:
        self.n_ch = n_ch
        self.capacity = capacity
        self._buf = np.zeros((capacity, n_ch), dtype=np.float64)
        self._total = 0
        self._lock = threading.Lock()

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    def push(self, x) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_ch:
            raise ValueError(f"expected (n, {self.n_ch}) samples, got {x.shape}")
        n = len(x)
        if n == 0:
            return
        with self._lock:
            # A push larger than the buffer only keeps its newest `capacity` samples,
            # but the absolute counter still advances by the full n.
            keep = min(n, self.capacity)
            start = self._total + (n - keep)
            self._buf[(start + np.arange(keep)) % self.capacity] = x[n - keep:]
            self._total += n

    def read_since(self, start: int):
        """Return (data, next_cursor, lost). `lost` counts samples that were overwritten unread."""
        with self._lock:
            total, cap = self._total, self.capacity
            oldest = max(0, total - cap)
            lost = 0
            if start < oldest:
                lost = oldest - start
                start = oldest
            start = min(start, total)  # defensive: cursor from a replaced buffer
            idx = np.arange(start, total) % cap
            return self._buf[idx], total, lost

    def latest(self, n: int) -> np.ndarray:
        """The most recent min(n, available) samples, oldest first."""
        with self._lock:
            total, cap = self._total, self.capacity
            n = min(n, cap, total)
            idx = np.arange(total - n, total) % cap
            return self._buf[idx]


class RateMeter:
    """Measured sample rate from arrival times, over a sliding window."""

    def __init__(self, window_s: float = 4.0) -> None:
        self.window_s = window_s
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._events: collections.deque = collections.deque()  # (arrival_time, cumulative_samples)
            self._count = 0

    def add(self, n: int, t: float | None = None) -> None:
        t = time.monotonic() if t is None else t
        with self._lock:
            self._count += n
            self._events.append((t, self._count))
            while len(self._events) > 2 and t - self._events[0][0] > self.window_s:
                self._events.popleft()

    @property
    def hz(self) -> float:
        with self._lock:
            if len(self._events) < 2:
                return 0.0
            (t0, c0), (t1, c1) = self._events[0], self._events[-1]
            if t1 - t0 < 0.5:
                return 0.0
            return (c1 - c0) / (t1 - t0)


class PacketStats:
    """Estimate dropped samples from gaps in PackNum.

    Assumes PackNum advances by 1 per sample. Whether Flex8 does that is verified by probe.py, so
    treat `lost` as an estimate. Big jumps and decreases are counted as counter resets, not losses.
    """

    MAX_PLAUSIBLE_GAP = 1000

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._prev: int | None = None
            self.received = 0
            self.lost = 0
            self.resets = 0

    def add(self, pack_nums) -> None:
        pack = np.asarray(pack_nums, dtype=np.int64)
        if len(pack) == 0:
            return
        with self._lock:
            seq = pack if self._prev is None else np.concatenate(([self._prev], pack))
            d = np.diff(seq)
            gap = (d > 1) & (d < self.MAX_PLAUSIBLE_GAP)
            self.lost += int((d[gap] - 1).sum())
            self.resets += int(((d < 0) | (d >= self.MAX_PLAUSIBLE_GAP)).sum())
            self.received += len(pack)
            self._prev = int(pack[-1])


class DataHub:
    """Sink for a source's on_data: raw ring + rate/loss stats + optional recorder.

    Runs on the source's thread. Never touches Qt.
    """

    def __init__(self, seconds: float = 60.0) -> None:
        self.seconds = seconds
        self.ring: RingBuffer | None = None
        self.channels: list = []
        self.fs = 250.0
        self.rate = RateMeter()
        self.stats = PacketStats()
        self.recorder = None
        self._lock = threading.Lock()

    def configure(self, channels, fs: float) -> None:
        with self._lock:
            n, cap = len(channels), int(round(self.seconds * fs))
            if self.ring is None or self.ring.n_ch != n or self.ring.capacity != cap:
                self.ring = RingBuffer(n, cap)
            self.channels = list(channels)
            self.fs = float(fs)
        self.rate.reset()
        self.stats.reset()

    def push(self, pack, marker, uv) -> None:
        ring = self.ring
        if ring is None or uv.shape[1] != ring.n_ch:
            return
        ring.push(uv)
        self.rate.add(len(uv))
        self.stats.add(pack)
        rec = self.recorder
        if rec is not None:
            rec.push(pack, marker, uv, time.time())
