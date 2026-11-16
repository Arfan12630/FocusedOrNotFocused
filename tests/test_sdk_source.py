"""Exercises SDKSource's whole worker flow against a fake pyneurosdk2 (no headset, no Bluetooth)."""
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest
from neurosdk.cmn_types import (EEGChannelId, EEGChannelType, SensorCommand, SensorFamily,
                                SensorSamplingFrequency, SensorState)

from brainviz import amp_raw
from brainviz.sdk_source import SDKSource
from brainviz.source import State

from helpers import wait_until
from test_amp_raw import FakeLib

# Four electrodes at scrambled SDK indices; sample j of each packet carries (j + 1) microvolts.
NUMS = [5, 2, 7, 0]
EXPECTED_ROW = [1.0, 3.0, 6.0, 8.0]  # columns follow sorted Num: 0, 2, 5, 7


class FakeSensor:
    instances = []

    def __init__(self, info):
        self.info = info
        self.sensor_ptr = None
        self.channels_count = 4
        self.commands = []
        self.disconnected = False
        self.signalDataReceived = self.resistDataReceived = None
        self.sensorStateChanged = self.batteryChanged = self.sensorAmpModeChanged = None
        self.version = NS(FwMajor=101, FwMinor=2, FwPatch=3, HwMajor=1, HwMinor=0, HwPatch=0)
        self.sampling_frequency = SensorSamplingFrequency.FrequencyHz250
        self.batt_power = 77
        self.amp_mode = NS(name="Idle")
        self.supported_channels = [
            NS(Num=n, Id=EEGChannelId.EEGChIdUnknown, Name="", ChType=EEGChannelType.EEGChTypeDifferential)
            for n in NUMS]
        self._stop = threading.Event()
        FakeSensor.instances.append(self)

    def is_supported_command(self, cmd):
        return True

    def exec_command(self, cmd):
        self.commands.append(cmd)
        if cmd is SensorCommand.StartSignal:
            self._pump("signalDataReceived", lambda i: [NS(PackNum=i * 5 + k, Marker=0,
                                                           Samples=[(j + 1) * 1e-6 for j in range(8)])
                                                        for k in range(5)])
        elif cmd is SensorCommand.StartResist:
            self._pump("resistDataReceived", lambda i: [NS(PackNum=i, Samples=[1e5 * (j + 1) for j in range(8)],
                                                           Referents=[])])
        elif cmd in (SensorCommand.StopSignal, SensorCommand.StopResist):
            self._stop.set()

    def _pump(self, attr, make):
        self._stop.set()
        self._stop = stop = threading.Event()

        def run():
            i = 0
            while not stop.is_set():
                cb = getattr(self, attr)
                if cb:
                    cb(self, make(i))
                i += 1
                time.sleep(0.02)

        threading.Thread(target=run, daemon=True).start()

    def disconnect(self):
        self.disconnected = True
        self._stop.set()


class FakeScanner:
    infos: list = []

    def __init__(self, filters):
        self.filters = filters
        self.sensorsChanged = None

    def start(self):
        if self.infos:  # native scanners report from their own thread
            threading.Timer(0.05, lambda: self.sensorsChanged(self, list(self.infos))).start()

    def stop(self):
        pass

    def sensors(self):
        return list(self.infos)

    def create_sensor(self, info):
        return FakeSensor(info)


def device(name="Flex Pro", rssi=-50, family=SensorFamily.LEBrainBitFlex, addr="AA:01"):
    return NS(SensFamily=family, SensModel=3, Name=name, Address=addr, SerialNumber="123456", RSSI=rssi,
              PairingRequired=False)


@pytest.fixture
def fake_sdk(monkeypatch):
    FakeSensor.instances = []
    FakeScanner.infos = [device()]
    lib = FakeLib(modes=0, gains=0, resist=[0] * 8, current=2)
    monkeypatch.setattr("neurosdk.scanner.Scanner", FakeScanner)
    monkeypatch.setattr("neurosdk.brainbit_2_sensor.BrainBit2Sensor", FakeSensor)
    monkeypatch.setattr(amp_raw, "_neuro_lib", lib)
    return lib


def run(gain=3, **kw):
    src = SDKSource(gain=gain, **kw)
    rec = NS(states=[], chunks=[], imps=[], ready=[])
    src.on_status = lambda s: rec.states.append(s)
    src.on_data = lambda p, m, uv: rec.chunks.append((p, m, uv))
    src.on_impedance = rec.imps.append
    src.on_ready = lambda ch, fs: rec.ready.append((ch, fs))
    src.start()
    return src, rec


def test_connect_configures_amplifier_streams_and_maps_channels(fake_sdk):
    src, rec = run()
    wait_until(lambda: rec.states and rec.states[-1].state is State.STREAMING, 5, "STREAMING")
    wait_until(lambda: len(rec.chunks) >= 3, 3, "samples")
    sensor = FakeSensor.instances[0]
    channels, fs = rec.ready[0]
    assert [c.num for c in channels] == [0, 2, 5, 7] and fs == 250.0
    assert [c.label for c in channels] == ["Ch1", "Ch3", "Ch6", "Ch8"]
    pack, marker, uv = rec.chunks[0]
    np.testing.assert_allclose(uv[0], EXPECTED_ROW)
    assert uv.shape[1] == 4 and len(pack) == 5
    assert rec.states[-1].battery == 77 and rec.states[-1].amp_mode == "Idle"
    # amplifier written (Normal mode, Gain3) for the 4 real channels, Current untouched
    assert list(fake_sdk.block.ChSignalMode)[:4] == [1] * 4 and list(fake_sdk.block.ChGain)[:4] == [2] * 4
    assert fake_sdk.block.Current == 2
    assert sensor.commands[:2] == [SensorCommand.Idle, SensorCommand.StartSignal]  # Idle before configuring
    assert src.info["firmware"] == "101.2.3" and src.info["gain"] == 3
    src.stop()
    assert sensor.disconnected and SensorCommand.StopSignal in sensor.commands


def test_gain_none_leaves_the_amplifier_alone(fake_sdk):
    src, rec = run(gain=None)
    wait_until(lambda: rec.states and rec.states[-1].state is State.STREAMING, 5, "STREAMING")
    src.stop()
    assert fake_sdk.writes == 0 and SensorCommand.Idle not in FakeSensor.instances[0].commands


def test_amplifier_failure_is_a_warning_not_a_dead_end(fake_sdk):
    def boom(*a):
        raise RuntimeError("amp refused")
    fake_sdk.writeAmplifierParamBrainBit2 = boom
    src, rec = run()
    wait_until(lambda: rec.states and rec.states[-1].state is State.STREAMING, 5, "STREAMING")
    assert "amplifier config failed" in rec.states[-1].message and "amp refused" in rec.states[-1].message
    wait_until(lambda: rec.chunks, 3, "samples despite the failure")
    src.stop()


def test_impedance_round_trip_pauses_and_resumes_signal(fake_sdk):
    src, rec = run()
    wait_until(lambda: rec.states and rec.states[-1].state is State.STREAMING, 5, "STREAMING")
    src.start_impedance()
    wait_until(lambda: rec.states[-1].state is State.IMPEDANCE, 3, "IMPEDANCE")
    wait_until(lambda: rec.imps, 3, "impedance packets")
    np.testing.assert_allclose(rec.imps[-1], [1e5, 3e5, 6e5, 8e5])  # mapped through Num like signal
    src.stop_impedance()
    wait_until(lambda: rec.states[-1].state is State.STREAMING, 3, "STREAMING again")
    cmds = FakeSensor.instances[0].commands
    order = [c for c in cmds if c in (SensorCommand.StopSignal, SensorCommand.StartResist,
                                      SensorCommand.StopResist, SensorCommand.StartSignal)]
    assert order == [SensorCommand.StartSignal, SensorCommand.StopSignal, SensorCommand.StartResist,
                     SensorCommand.StopResist, SensorCommand.StartSignal]
    n = len(rec.chunks)
    wait_until(lambda: len(rec.chunks) > n, 3, "signal resumed")
    src.stop()


def test_impedance_unavailable_falls_back_to_signal(fake_sdk, monkeypatch):
    src, rec = run()
    wait_until(lambda: rec.states and rec.states[-1].state is State.STREAMING, 5, "STREAMING")
    sensor = FakeSensor.instances[0]
    real = sensor.exec_command

    def refuse(cmd):
        if cmd is SensorCommand.StartResist:
            raise RuntimeError("not in this amp mode")
        real(cmd)
    sensor.exec_command = refuse
    src.start_impedance()
    wait_until(lambda: "Impedance check unavailable" in rec.states[-1].message, 3, "fallback message")
    assert rec.states[-1].state is State.STREAMING
    src.stop()


def test_connection_loss_then_reconnect(fake_sdk):
    src, rec = run()
    wait_until(lambda: rec.states and rec.states[-1].state is State.STREAMING, 5, "STREAMING")
    first = FakeSensor.instances[0]
    first.sensorStateChanged(first, SensorState.StateOutOfRange)  # as the native layer would
    wait_until(lambda: rec.states[-1].state is State.DISCONNECTED, 2, "DISCONNECTED")
    src.reconnect()
    wait_until(lambda: len(FakeSensor.instances) == 2 and rec.states[-1].state is State.STREAMING, 5, "reconnect")
    assert first.disconnected
    src.stop()


def test_no_device_gives_actionable_error(fake_sdk):
    FakeScanner.infos = []
    src, rec = run(scan_timeout=0.6)
    wait_until(lambda: rec.states and rec.states[-1].state is State.ERROR, 5, "ERROR")
    msg = rec.states[-1].message
    assert "NeuroREC" in msg and "Bluetooth permission" in msg
    src.stop()


def test_prefers_flex_over_other_families_then_strongest_signal(fake_sdk):
    FakeScanner.infos = [device("Pro", -30, SensorFamily.LEBrainBitPro, "AA:02"),
                         device("Flex weak", -80, SensorFamily.LEBrainBitFlex, "AA:03"),
                         device("Flex strong", -40, SensorFamily.LEBrainBitFlex, "AA:04")]
    src, rec = run()
    wait_until(lambda: rec.states and rec.states[-1].state is State.STREAMING, 5, "STREAMING")
    assert FakeSensor.instances[0].info.Name == "Flex strong"
    src.stop()


def test_malformed_packets_are_reported_once_and_do_not_kill_the_stream(fake_sdk):
    src, rec = run()
    wait_until(lambda: rec.states and rec.states[-1].state is State.STREAMING, 5, "STREAMING")
    sensor = FakeSensor.instances[0]
    for _ in range(3):
        sensor.signalDataReceived(sensor, [NS(PackNum=0, Marker=0, Samples=[1e-6] * 3)])  # too short for Num 7
    errors = [s for s in rec.states if s.state is State.ERROR]
    assert len(errors) == 1 and "Unusable signal packet" in errors[0].message
    n = len(rec.chunks)
    wait_until(lambda: len(rec.chunks) > n, 3, "valid packets still flow")
    src.stop()
