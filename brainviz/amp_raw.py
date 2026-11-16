"""Raw amplifier-parameter access for BrainBit2-family sensors (Flex8 included).

Why this exists: pyneurosdk2 1.0.15 declares
    class BrainBit2ChannelMode(OrderedEnum):
        ChModeShort = 0,        # <- trailing comma: the value is the tuple (0,)
so `BrainBit2ChannelMode(0)` raises ValueError. The `amplifier_param` property decodes every channel's
mode through that enum, so reading it fails whenever any channel is in Short mode, and writing
ChModeShort fails too. We call the native read/write directly, change only the signal-mode and gain
bytes, and carry `Current` and `ChResistUse` through untouched (their enums are also unreliable).

This depends on private wrapper internals, hence the exact pin pyneurosdk2==1.0.15.
"""
from __future__ import annotations

import ctypes

from neurosdk.__cmn_types import BRAINBIT2_MAX_CH_COUNT, NativeBrainBit2AmplifierParam, OpStatus
from neurosdk.__utils import raise_exception_if
from neurosdk.cmn_types import SensorGain
from neurosdk.neuro_lib_load import _neuro_lib

MODE_SHORT = 0   # shorted input
MODE_NORMAL = 1  # bipolar input, used for EEG


def gain_value(gain: int) -> int:
    """Map 1, 2, 3, 4, 6, 8, 12, 24 (or 5) to the SDK's SensorGain byte value."""
    try:
        return SensorGain[f"Gain{int(gain)}"].value
    except KeyError:
        allowed = sorted(int(m.name[4:]) for m in SensorGain if m.name.startswith("Gain") and m.name[4:].isdigit())
        raise ValueError(f"unsupported gain {gain!r}; choose one of {allowed}") from None


def read_raw(sensor, lib=None) -> NativeBrainBit2AmplifierParam:
    lib = lib or _neuro_lib
    params = NativeBrainBit2AmplifierParam()
    status = OpStatus()
    lib.readAmplifierParamBrainBit2(sensor.sensor_ptr, ctypes.pointer(params), ctypes.pointer(status))
    raise_exception_if(status)
    return params


def write_raw(sensor, params: NativeBrainBit2AmplifierParam, lib=None) -> None:
    lib = lib or _neuro_lib
    status = OpStatus()
    lib.writeAmplifierParamBrainBit2(sensor.sensor_ptr, params, ctypes.pointer(status))
    raise_exception_if(status)


def snapshot(params: NativeBrainBit2AmplifierParam, n_channels: int) -> dict:
    """Plain-int view of the parameter block (no enum decoding)."""
    n = min(n_channels, BRAINBIT2_MAX_CH_COUNT)
    return {
        "ChSignalMode": [int(params.ChSignalMode[i]) for i in range(n)],
        "ChResistUse": [int(params.ChResistUse[i]) for i in range(n)],
        "ChGain": [int(params.ChGain[i]) for i in range(n)],
        "Current": int(params.Current),
    }


def configure(sensor, gain: int, n_channels: int | None = None, lib=None) -> tuple[dict, dict]:
    """Set every channel to Normal (bipolar) mode with the given gain. Returns (before, after).

    The sensor must be in Idle or PowerDown amp mode.
    """
    n = n_channels if n_channels is not None else int(sensor.channels_count)
    value = gain_value(gain)
    params = read_raw(sensor, lib)
    before = snapshot(params, n)
    for i in range(min(n, BRAINBIT2_MAX_CH_COUNT)):
        params.ChSignalMode[i] = MODE_NORMAL
        params.ChGain[i] = value
    write_raw(sensor, params, lib)
    after = snapshot(read_raw(sensor, lib), n)
    return before, after
