import ctypes
from types import SimpleNamespace as NS

import pytest
from neurosdk.__cmn_types import NativeBrainBit2AmplifierParam
from neurosdk.cmn_types import BrainBit2ChannelMode, SensorGain

from brainviz import amp_raw


class FakeLib:
    """Stands in for the native library: reads/writes a parameter block held in memory."""

    def __init__(self, modes, gains, resist, current):
        self.block = NativeBrainBit2AmplifierParam()
        for i in range(8):
            self.block.ChSignalMode[i], self.block.ChGain[i], self.block.ChResistUse[i] = modes, gains, resist[i]
        self.block.Current = current
        self.writes = 0

    def readAmplifierParamBrainBit2(self, ptr, out, status):
        ctypes.memmove(ctypes.addressof(out.contents), ctypes.addressof(self.block), ctypes.sizeof(self.block))

    def writeAmplifierParamBrainBit2(self, ptr, params, status):
        ctypes.memmove(ctypes.addressof(self.block), ctypes.addressof(params), ctypes.sizeof(self.block))
        self.writes += 1


SENSOR = NS(sensor_ptr=None, channels_count=8)


def test_configure_sets_mode_and_gain_and_preserves_everything_else():
    lib = FakeLib(modes=0, gains=0, resist=[1, 0, 1, 0, 1, 0, 1, 0], current=3)
    before, after = amp_raw.configure(SENSOR, 3, lib=lib)
    assert before["ChSignalMode"] == [0] * 8  # Short: the value the SDK's own enum cannot decode
    assert after["ChSignalMode"] == [amp_raw.MODE_NORMAL] * 8
    assert after["ChGain"] == [SensorGain.Gain3.value] * 8
    assert after["ChResistUse"] == [1, 0, 1, 0, 1, 0, 1, 0]  # untouched
    assert after["Current"] == 3  # untouched
    assert lib.writes == 1


def test_only_the_first_n_channels_are_changed():
    lib = FakeLib(modes=0, gains=0, resist=[0] * 8, current=0)
    _, after = amp_raw.configure(NS(sensor_ptr=None, channels_count=4), 6, n_channels=4, lib=lib)
    assert after["ChGain"] == [SensorGain.Gain6.value] * 4
    assert list(lib.block.ChGain) == [SensorGain.Gain6.value] * 4 + [0] * 4


@pytest.mark.parametrize("gain,member", [(1, "Gain1"), (3, "Gain3"), (6, "Gain6"), (24, "Gain24")])
def test_gain_value_matches_sdk_enum(gain, member):
    assert amp_raw.gain_value(gain) == SensorGain[member].value


def test_unsupported_gain_lists_choices():
    with pytest.raises(ValueError, match="choose one of"):
        amp_raw.gain_value(7)


def test_documents_the_sdk_enum_bug_this_module_works_around():
    """If a newer pyneurosdk2 fixes ChModeShort (trailing comma), this fails and the pin can be revisited."""
    assert BrainBit2ChannelMode.ChModeShort.value == (0,)
    with pytest.raises(ValueError):
        BrainBit2ChannelMode(0)
