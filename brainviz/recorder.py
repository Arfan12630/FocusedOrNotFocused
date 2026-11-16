"""CSV recording on its own thread, so a stalled UI or slow disk never costs samples."""
from __future__ import annotations

import datetime as dt
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np

CSV_NOTE = (
    "Channel columns are microvolts, unfiltered (DC and mains included). "
    "host_time_unix is when the packet reached this computer, not the ADC sample time; "
    "samples inside one packet share it. sample_index counts samples since recording start."
)


class Recorder:
    def __init__(self, out_dir, labels: list[str], fs: float, meta: dict | None = None,
                 simulated: bool = False) -> None:
        self.out_dir = Path(out_dir)
        self.labels = list(labels)
        self.fs = fs
        self.meta = dict(meta or {})
        self.simulated = simulated
        self.path: Path | None = None
        self.sidecar: Path | None = None
        self.samples_written = 0
        self.error: str | None = None
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        self._t0 = 0.0
        self._started_at: dt.datetime | None = None

    # -- control -----------------------------------------------------------
    def start(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S") + ("-SIM" if self.simulated else "")
        path = self.out_dir / f"{stamp}.csv"
        n = 2
        while path.exists():
            path = self.out_dir / f"{stamp}-{n}.csv"
            n += 1
        self.path = path
        self.sidecar = path.with_suffix(".json")
        self._started_at = dt.datetime.now().astimezone()
        self._t0 = time.monotonic()
        self._write_sidecar({"state": "recording"})
        self._running = True
        self._thread = threading.Thread(target=self._writer, name="csv-writer", daemon=True)
        self._thread.start()
        return path

    def push(self, pack, marker, uv, host_time: float) -> None:
        if self._running:
            self._q.put((pack, marker, uv, host_time))

    def stop(self, extra: dict | None = None) -> dict:
        """Flush, close and finalise the sidecar. Returns a summary dict."""
        if self._thread is None:
            return {}
        self._running = False
        self._q.put(None)
        self._thread.join(timeout=10)
        self._thread = None
        duration = time.monotonic() - self._t0
        summary = {
            "state": "complete" if self.error is None else "error",
            "csv": self.path.name,
            "samples": self.samples_written,
            "duration_s": round(duration, 3),
            "stopped_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        if self.error:
            summary["error"] = self.error
        summary.update(extra or {})
        self._write_sidecar(summary)
        return {"path": self.path, "samples": self.samples_written, "duration_s": duration, **(extra or {})}

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._t0 if self._running else 0.0

    # -- internals -----------------------------------------------------------
    def _write_sidecar(self, extra: dict) -> None:
        doc = {
            "csv": self.path.name,
            "started_at": self._started_at.isoformat(timespec="seconds"),
            "simulated": self.simulated,
            "fs_hz": self.fs,
            "units": "microvolts",
            "columns": ["sample_index", "host_time_unix", "pack_num", "marker", *self.labels],
            "note": CSV_NOTE,
            **self.meta,
            **extra,
        }
        tmp = self.sidecar.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2, default=str))
        tmp.replace(self.sidecar)

    def _writer(self) -> None:
        try:
            with open(self.path, "w", newline="", buffering=1 << 16) as f:
                f.write(",".join(_csv_field(c) for c in
                                 ["sample_index", "host_time_unix", "pack_num", "marker", *self.labels]) + "\n")
                last_flush = time.monotonic()
                index = 0
                while True:
                    item = self._q.get()
                    if item is None:
                        break
                    pack, marker, uv, host_time = item
                    n = len(uv)
                    lines = []
                    for i in range(n):
                        row = ",".join(f"{v:.3f}" for v in uv[i])
                        lines.append(f"{index + i},{host_time:.6f},{int(pack[i])},{int(marker[i])},{row}")
                    f.write("\n".join(lines) + "\n")
                    index += n
                    self.samples_written = index
                    if time.monotonic() - last_flush > 1.0:
                        f.flush()
                        last_flush = time.monotonic()
        except Exception as exc:  # disk full, permissions, ...
            self.error = f"{type(exc).__name__}: {exc}"
            self._running = False


def _csv_field(text: str) -> str:
    return f'"{text}"' if any(c in text for c in ',"\n') else text


def read_csv(path) -> tuple[list[str], np.ndarray]:
    """Load a recording (header names, data array). Used by tests and for quick analysis."""
    path = Path(path)
    with open(path) as f:
        header = f.readline().rstrip("\n").split(",")
    return header, np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
