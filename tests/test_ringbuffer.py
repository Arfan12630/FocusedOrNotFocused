import numpy as np

from brainviz.ringbuffer import DataHub, PacketStats, RateMeter, RingBuffer
from brainviz.source import Channel


def seq(start, n, n_ch=2):
    """Samples whose value encodes their absolute index (channel 1 is index + 0.5)."""
    idx = np.arange(start, start + n, dtype=float)
    return np.stack([idx, idx + 0.5], axis=1)[:, :n_ch]


def test_push_and_read_in_order():
    rb = RingBuffer(2, 10)
    rb.push(seq(0, 4))
    data, cur, lost = rb.read_since(0)
    assert lost == 0 and cur == 4
    np.testing.assert_array_equal(data[:, 0], [0, 1, 2, 3])
    data, cur, lost = rb.read_since(cur)
    assert len(data) == 0 and cur == 4


def test_wraparound_keeps_order():
    rb = RingBuffer(2, 10)
    rb.push(seq(0, 8))
    _, cur, _ = rb.read_since(0)
    rb.push(seq(8, 6))  # wraps past the end of the storage
    data, cur2, lost = rb.read_since(cur)
    assert lost == 0 and cur2 == 14
    np.testing.assert_array_equal(data[:, 0], np.arange(8, 14))
    np.testing.assert_array_equal(rb.latest(10)[:, 0], np.arange(4, 14))


def test_overrun_reports_lost_samples():
    rb = RingBuffer(2, 10)
    rb.push(seq(0, 25))  # reader never read; only the newest 10 survive
    data, cur, lost = rb.read_since(0)
    assert lost == 15 and cur == 25
    np.testing.assert_array_equal(data[:, 0], np.arange(15, 25))


def test_oversized_single_push_advances_counter_fully():
    rb = RingBuffer(2, 10)
    rb.push(seq(0, 3))
    rb.push(seq(3, 50))
    assert rb.total == 53
    np.testing.assert_array_equal(rb.latest(10)[:, 0], np.arange(43, 53))


def test_latest_is_bounded_and_empty_safe():
    rb = RingBuffer(2, 10)
    assert rb.latest(5).shape == (0, 2)
    rb.push(seq(0, 3))
    assert rb.latest(100).shape == (3, 2)


def test_cursor_from_a_replaced_buffer_is_clamped():
    rb = RingBuffer(2, 10)
    rb.push(seq(0, 5))
    data, cur, lost = rb.read_since(9999)
    assert len(data) == 0 and cur == 5 and lost == 0


def test_rate_meter_measures_hz():
    rm = RateMeter(window_s=10)
    t = 100.0
    for _ in range(40):  # 10 samples every 40 ms = 250 Hz
        rm.add(10, t)
        t += 0.04
    assert abs(rm.hz - 250) < 5


def test_packet_stats_counts_gaps_and_resets():
    ps = PacketStats()
    ps.add([1, 2, 3])
    ps.add([4, 5])
    assert ps.lost == 0 and ps.received == 5
    ps.add([9, 10])  # 6,7,8 missing across the call boundary
    assert ps.lost == 3
    ps.add([2, 3])  # counter went backwards: a reset, not a loss
    assert ps.lost == 3 and ps.resets == 1


def test_hub_ignores_data_until_configured_and_feeds_recorder():
    hub = DataHub(seconds=2)
    uv = seq(0, 5)
    hub.push(np.arange(5), np.zeros(5, int), uv)  # not configured yet: dropped, no exception
    hub.configure([Channel(0, "A"), Channel(1, "B")], 250.0)
    got = []
    hub.recorder = type("R", (), {"push": lambda self, *a: got.append(a)})()
    hub.push(np.arange(5), np.zeros(5, int), uv)
    assert hub.ring.total == 5 and len(got) == 1
    assert hub.ring.capacity == 500
