"""Real BrainBit source built on pyneurosdk2.

One worker thread owns every blocking SDK call (scan, create_sensor, exec_command, disconnect),
driven by a command queue. Native callbacks (signal, resist, state, battery) only convert data and
emit events; they never call back into the SDK.
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

from .source import EEGSource, State, make_channels, packets_to_array

# Preference order when several devices are in range; Flex first. All three map to BrainBit2Sensor.
SCAN_FAMILIES = ("LEBrainBitFlex", "LEBrainBitPro", "LEBrainBit2")
DEFAULT_FS = 250.0
NO_DATA_WARN_S = 5.0


def _try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


class SDKSource(EEGSource):
    def __init__(self, gain: int | None = 3, scan_timeout: float = 30.0) -> None:
        super().__init__()
        self.gain = gain  # None: leave the device's amplifier parameters alone
        self.scan_timeout = scan_timeout
        self._cmds: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._scanner = None
        self._sensor = None
        self._nums: list[int] = []
        self._samples_seen = 0
        self._signal_error_reported = False

    # -- public API (any thread) -------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="sdk-worker", daemon=True)
        self._thread.start()
        self._cmds.put("connect")

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_evt.set()
        self._cmds.put("shutdown")
        self._thread.join(timeout=8)
        self._thread = None

    def reconnect(self) -> None:
        self._cmds.put("reconnect")

    def start_impedance(self) -> None:
        self._cmds.put("impedance_start")

    def stop_impedance(self) -> None:
        self._cmds.put("impedance_stop")

    # -- worker thread --------------------------------------------------------
    def _run(self) -> None:
        while True:
            cmd = self._cmds.get()
            if cmd == "shutdown":
                break
            try:
                if cmd == "connect":
                    self._connect()
                elif cmd == "reconnect":
                    self._teardown()
                    self._connect()
                elif cmd == "impedance_start":
                    self._impedance_start()
                elif cmd == "impedance_stop":
                    self._impedance_stop()
            except Exception as exc:
                self._emit_status(State.ERROR, f"{cmd} failed: {exc}")
        self._teardown()
        self._emit_status(State.IDLE, "Stopped")

    def _connect(self) -> None:
        try:
            from neurosdk.brainbit_2_sensor import BrainBit2Sensor
            from neurosdk.cmn_types import SensorCommand, SensorFamily, SensorSamplingFrequency
            from neurosdk.scanner import Scanner
        except ImportError as exc:
            self._emit_status(State.ERROR, f"pyneurosdk2 is not installed or failed to load: {exc}")
            return

        self._emit_status(State.SCANNING, "Scanning for BrainBit Flex (close NeuroREC first)")
        found: dict = {}
        seen = threading.Event()

        def on_found(_scanner, sensors):
            for s in sensors:
                found[s.Address] = s
            if sensors:
                seen.set()

        families = [getattr(SensorFamily, name) for name in SCAN_FAMILIES]
        self._scanner = Scanner(families)
        self._scanner.sensorsChanged = on_found
        self._scanner.start()
        t0 = time.monotonic()
        while not self._stop_evt.is_set() and time.monotonic() - t0 < self.scan_timeout:
            if seen.wait(0.2):
                time.sleep(1.0)  # let other devices show up so we can pick the strongest
                break
        _try(self._scanner.stop)
        candidates = _try(self._scanner.sensors, []) or list(found.values())
        if self._stop_evt.is_set():
            return
        if not candidates:
            self._emit_status(State.ERROR, "No BrainBit device found. Check it is on, charged and not "
                              "connected to NeuroREC, and that this terminal app has Bluetooth permission.")
            return

        def rank(info):
            order = [getattr(SensorFamily, n) for n in SCAN_FAMILIES]
            return (order.index(info.SensFamily) if info.SensFamily in order else 99, -info.RSSI)

        info = sorted(candidates, key=rank)[0]
        self._emit_status(State.CONNECTING, f"Connecting to {info.Name} ({info.Address})")
        sensor = self._scanner.create_sensor(info)  # blocking
        if not isinstance(sensor, BrainBit2Sensor):
            self._emit_status(State.ERROR, f"{info.Name} is a {type(sensor).__name__}, expected BrainBit2Sensor")
            return
        self._sensor = sensor
        sensor.sensorStateChanged = self._on_state
        sensor.batteryChanged = self._on_battery
        sensor.sensorAmpModeChanged = self._on_amp_mode

        version = _try(lambda: sensor.version)
        self.info = {
            "device_name": info.Name,
            "address": info.Address,
            "serial": info.SerialNumber,
            "family": info.SensFamily.name,
            "model": info.SensModel,
            "firmware": f"{version.FwMajor}.{version.FwMinor}.{version.FwPatch}" if version else None,
            "hardware": f"{version.HwMajor}.{version.HwMinor}.{version.HwPatch}" if version else None,
            "rssi": info.RSSI,
        }
        freq = _try(lambda: sensor.sampling_frequency)
        fs = DEFAULT_FS
        if freq is not None and freq != SensorSamplingFrequency.FrequencyUnsupported:
            fs = _try(lambda: float(freq.name.replace("FrequencyHz", "")), DEFAULT_FS)
        battery = _try(lambda: sensor.batt_power)

        channels = make_channels(sensor.supported_channels, self.label_overrides)
        self._nums = [c.num for c in channels]

        warnings = []
        if self.gain is not None:
            from . import amp_raw
            if sensor.is_supported_command(SensorCommand.Idle):
                _try(lambda: sensor.exec_command(SensorCommand.Idle))
            try:
                before, after = amp_raw.configure(sensor, self.gain, n_channels=sensor.channels_count)
                self.info["amplifier_before"], self.info["amplifier_after"] = before, after
                self.info["gain"] = self.gain
            except Exception as exc:
                warnings.append(f"amplifier config failed ({exc}); continuing with device defaults")
        amp_mode = _try(lambda: sensor.amp_mode.name, "")

        self._samples_seen = 0
        self._signal_error_reported = False
        self._publish_ready(channels, fs)
        sensor.signalDataReceived = self._on_signal
        sensor.exec_command(SensorCommand.StartSignal)
        message = f"{info.Name} connected, {len(channels)} channels @ {fs:g} Hz"
        if warnings:
            message += "; " + "; ".join(warnings)
        self._emit_status(State.STREAMING, message, battery=battery, amp_mode=amp_mode)
        threading.Timer(NO_DATA_WARN_S, self._check_data_flowing, args=(sensor,)).start()

    def _check_data_flowing(self, sensor) -> None:
        if sensor is self._sensor and self._samples_seen == 0 and self.status.state is State.STREAMING:
            self._emit_status(State.STREAMING, f"Connected but no samples after {NO_DATA_WARN_S:g} s. "
                              "Run probe.py to inspect the amplifier configuration.")

    def _impedance_start(self) -> None:
        from neurosdk.cmn_types import SensorCommand
        sensor = self._sensor
        if sensor is None or self.status.state is not State.STREAMING:
            return
        sensor.exec_command(SensorCommand.StopSignal)
        sensor.resistDataReceived = self._on_resist
        try:
            sensor.exec_command(SensorCommand.StartResist)
        except Exception as exc:
            sensor.resistDataReceived = None
            sensor.exec_command(SensorCommand.StartSignal)
            self._emit_status(State.STREAMING, f"Impedance check unavailable: {exc}")
            return
        self._emit_status(State.IMPEDANCE, "Measuring impedance (signal paused)")

    def _impedance_stop(self) -> None:
        from neurosdk.cmn_types import SensorCommand
        sensor = self._sensor
        if sensor is None or self.status.state is not State.IMPEDANCE:
            return
        _try(lambda: sensor.exec_command(SensorCommand.StopResist))
        sensor.resistDataReceived = None
        sensor.exec_command(SensorCommand.StartSignal)
        self._emit_status(State.STREAMING, "Signal resumed")

    def _teardown(self) -> None:
        sensor, scanner = self._sensor, self._scanner
        self._sensor = self._scanner = None
        if sensor is not None:
            from neurosdk.cmn_types import SensorCommand
            for step in (
                lambda: setattr(sensor, "signalDataReceived", None),
                lambda: setattr(sensor, "resistDataReceived", None),
                lambda: setattr(sensor, "sensorStateChanged", None),
                lambda: setattr(sensor, "batteryChanged", None),
                lambda: setattr(sensor, "sensorAmpModeChanged", None),
                lambda: sensor.exec_command(SensorCommand.StopSignal),
                lambda: sensor.disconnect(),
            ):
                _try(step)
        del sensor
        del scanner
        import gc
        gc.collect()

    # -- native callbacks (SDK threads): convert and emit only ---------------
    def _on_signal(self, _sensor, data) -> None:
        try:
            pack, marker, uv = packets_to_array(data, self._nums)
        except Exception as exc:
            if not self._signal_error_reported:
                self._signal_error_reported = True
                self._emit_status(State.ERROR, f"Unusable signal packet: {exc}")
            return
        self._samples_seen += len(uv)
        cb = self.on_data
        if cb is not None:
            cb(pack, marker, uv)

    def _on_resist(self, _sensor, data) -> None:
        if not data or self.on_impedance is None:
            return
        try:
            ohms = np.asarray(data[-1].Samples, dtype=np.float64)[self._nums]
        except Exception:
            return
        self.on_impedance(ohms)

    def _on_state(self, _sensor, state) -> None:
        from neurosdk.cmn_types import SensorState
        if state == SensorState.StateOutOfRange:
            self._emit_status(State.DISCONNECTED, "Connection lost (out of range, powered off, or taken by "
                              "another app). Press Reconnect.")

    def _on_battery(self, _sensor, battery) -> None:
        self._patch_status(battery=int(battery))

    def _on_amp_mode(self, _sensor, mode) -> None:
        self._patch_status(amp_mode=getattr(mode, "name", str(mode)))
