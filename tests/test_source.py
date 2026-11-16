from types import SimpleNamespace as NS

import numpy as np
import pytest
from neurosdk.cmn_types import EEGChannelId, EEGChannelType

from brainviz.source import Channel, apply_labels, make_channels, packets_to_array


def info(num, ch_id=EEGChannelId.EEGChIdUnknown, name=""):
    return NS(Num=num, Id=ch_id, Name=name, ChType=EEGChannelType.EEGChTypeDifferential)


def test_label_priority_name_then_id_then_numbered():
    chans = make_channels([
        info(0, EEGChannelId.EEGChIdO1, "Left"),   # SDK name wins
        info(1, EEGChannelId.EEGChIdT3, ""),       # id fallback, prefix stripped
        info(2, EEGChannelId.EEGChIdUnknown, ""),  # free electrode
        info(3, EEGChannelId.EEGChIdUnknown, "Unknown"),
    ])
    assert [c.label for c in chans] == ["Left", "T3", "Ch3", "Ch4"]


def test_flex8_reports_any_for_every_electrode_and_gets_numbered_labels():
    # Observed on a real Flex8: Id=Unknown, Type=Ref, Name='Any' on all eight channels.
    chans = make_channels([info(n, EEGChannelId.EEGChIdUnknown, "Any") for n in range(8)])
    assert [c.label for c in chans] == [f"Ch{n + 1}" for n in range(8)]


def test_channels_sorted_by_num_and_keep_num():
    chans = make_channels([info(5), info(0), info(7), info(2)])
    assert [c.num for c in chans] == [0, 2, 5, 7]
    assert [c.label for c in chans] == ["Ch1", "Ch3", "Ch6", "Ch8"]


def test_duplicate_labels_are_made_unique():
    chans = make_channels([info(0, EEGChannelId.EEGChIdO1), info(1, EEGChannelId.EEGChIdO1)])
    assert len({c.label for c in chans}) == 2


def test_overrides_apply_by_position_and_tolerate_short_lists():
    base = [Channel(0, "a"), Channel(1, "b"), Channel(2, "c")]
    out = apply_labels(base, ["X", "", "Z", "extra"])
    assert [c.label for c in out] == ["X", "b", "Z"]
    assert [c.label for c in apply_labels(base, ["X"])] == ["X", "b", "c"]


def packet(pack, samples, marker=0):
    return NS(PackNum=pack, Marker=marker, Samples=samples)


def test_scrambled_num_maps_to_the_right_sample_and_converts_to_microvolts():
    # 4 supported channels living at SDK sample indices 0, 2, 5, 7 of an 8-wide packet.
    nums = [c.num for c in make_channels([info(5), info(0), info(7), info(2)])]
    volts = [(j + 1) * 1e-6 for j in range(8)]  # sample j is (j+1) microvolts
    pack, marker, uv = packets_to_array([packet(10, volts, 3), packet(11, volts)], nums)
    np.testing.assert_allclose(uv[0], [1, 3, 6, 8])  # columns follow the sorted Num order
    np.testing.assert_array_equal(pack, [10, 11])
    np.testing.assert_array_equal(marker, [3, 0])
    assert uv.shape == (2, 4)


def test_empty_and_malformed_packets():
    p, m, uv = packets_to_array([], [0, 1])
    assert uv.shape == (0, 2) and len(p) == 0
    with pytest.raises(ValueError, match="different sample counts"):
        packets_to_array([packet(0, [1e-6] * 8), packet(1, [1e-6] * 6)], [0])
    with pytest.raises(ValueError, match="Num 7"):
        packets_to_array([packet(0, [1e-6] * 4)], [0, 7])
