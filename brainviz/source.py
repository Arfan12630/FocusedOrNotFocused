"""Common EEG source interface and helpers shared by the SDK and simulated sources."""
from __future__ import annotations

import abc
import enum
import threading
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

V_TO_UV = 1e6  # the SDK delivers volts


class State(str, enum.Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    IMPEDANCE = "impedance"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass(frozen=True)
class Status:
    state: State
    message: str = ""
    battery: int | None = None
    amp_mode: str = ""


@dataclass(frozen=True)
class Channel:
    """One data column. `num` is the index into the SDK's Samples array."""

    num: int
    label: str
    sdk_id: str = ""
    sdk_name: str = ""


def _clean(text: str | None) -> str:
    text = (text or "").strip()
    # The Flex8 reports every electrode as Name="Any" (free placement), which is not a usable label.
    return "" if text.lower() in ("", "unknown", "any") else text


def make_channels(infos: Sequence, overrides: Sequence[str] | None = None) -> list[Channel]:
    """Build the channel list from the SDK's `supported_channels` (objects with Num/Id/Name).

    Label priority: SDK Name, then SDK Id (e.g. EEGChIdO1 -> O1), then Ch{Num+1}.
    Flex electrodes are free-placement, so "Unknown" is expected and falls through.
    Data columns are ordered by Num so that column i always matches sample index Num.
    """
    channels: list[Channel] = []
    for info in sorted(infos, key=lambda i: int(i.Num)):
        num = int(info.Num)
        ch_id = getattr(info.Id, "name", str(info.Id))
        id_label = _clean(ch_id[len("EEGChId"):] if ch_id.startswith("EEGChId") else ch_id)
        label = _clean(info.Name) or id_label or f"Ch{num + 1}"
        channels.append(Channel(num=num, label=label, sdk_id=ch_id, sdk_name=info.Name or ""))
    return apply_labels(channels, overrides)


def apply_labels(channels: Sequence[Channel], overrides: Sequence[str] | None = None) -> list[Channel]:
    """Apply user labels by position, then make every label unique."""
    out: list[Channel] = []
    seen: set[str] = set()
    for i, ch in enumerate(channels):
        label = ch.label
        if overrides and i < len(overrides) and overrides[i].strip():
            label = overrides[i].strip()
        if label in seen:
            label = f"{label}.{ch.num + 1}"
        seen.add(label)
        out.append(Channel(ch.num, label, ch.sdk_id, ch.sdk_name))
    return out


def packets_to_array(packets: Sequence, nums: Sequence[int]):
    """Convert SDK SignalChannelsData-like packets into arrays.

    Each packet is one time sample across all channels (PackNum, Marker, Samples in volts).
    Returns (pack_nums[int64, n], markers[int64, n], uv[float64, n x len(nums)]) where column i
    is Samples[nums[i]] converted to microvolts.
    """
    n = len(packets)
    nums = list(nums)
    if n == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty((0, len(nums)))
    pack = np.fromiter((p.PackNum for p in packets), dtype=np.int64, count=n)
    marker = np.fromiter((p.Marker for p in packets), dtype=np.int64, count=n)
    try:
        raw = np.array([p.Samples for p in packets], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("packets in one callback have different sample counts") from exc
    if raw.ndim != 2 or raw.shape[1] <= max(nums):
        raise ValueError(f"packet carries {raw.shape[-1]} samples but channel Num {max(nums)} is required")
    return pack, marker, raw[:, nums] * V_TO_UV


class EEGSource(abc.ABC):
    """A producer of EEG samples. Callbacks are invoked from non-GUI threads.

    Set the callbacks before start():
      on_ready(channels, fs)          channel table known; fired before the first sample
      on_data(pack, marker, uv)       arrays from packets_to_array (microvolts)
      on_status(Status)               connection / mode changes
      on_impedance(ohms)              ndarray (n_channels,) while in impedance mode
    """

    simulated = False

    def __init__(self) -> None:
        self.on_ready: Callable | None = None
        self.on_data: Callable | None = None
        self.on_status: Callable | None = None
        self.on_impedance: Callable | None = None
        self.fs: float = 250.0
        self.channels: list[Channel] = []
        self.label_overrides: list[str] = []
        self.info: dict = {}  # device facts for the recording sidecar
        self.status = Status(State.IDLE)
        self._status_lock = threading.Lock()

    # -- lifecycle -------------------------------------------------------
    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def reconnect(self) -> None: ...

    @abc.abstractmethod
    def start_impedance(self) -> None: ...

    @abc.abstractmethod
    def stop_impedance(self) -> None: ...

    # -- helpers for subclasses -----------------------------------------
    def _emit_status(self, state: State, message: str = "", battery: int | None = None,
                     amp_mode: str | None = None) -> None:
        with self._status_lock:
            prev = self.status
            self.status = Status(
                state=state,
                message=message,
                battery=battery if battery is not None else prev.battery,
                amp_mode=amp_mode if amp_mode is not None else prev.amp_mode,
            )
            snapshot = self.status
        if self.on_status:
            self.on_status(snapshot)

    def _patch_status(self, battery: int | None = None, amp_mode: str | None = None) -> None:
        """Update battery / amp mode while keeping the current state and message."""
        prev = self.status
        self._emit_status(prev.state, prev.message, battery=battery, amp_mode=amp_mode)

    def _publish_ready(self, channels: Sequence[Channel], fs: float) -> None:
        self.channels = apply_labels(channels, self.label_overrides)
        self.fs = float(fs)
        if self.on_ready:
            self.on_ready(self.channels, self.fs)
