#!/usr/bin/env python
"""Minimal hardware probe for the BrainBit Flex8: scan, connect, print a few seconds of samples.

Run from the project root, from the terminal app that has Bluetooth permission (iTerm), with the
headset on and NeuroREC closed:

    .venv/bin/python probe.py

It captures twice so you can see whether amplifier parameters must be set before data flows:
  phase A  StartSignal exactly as the device comes up
  phase B  Idle -> write Normal mode + gain on every channel -> StartSignal
Add --resist to also read impedance for a few seconds.
"""
from __future__ import annotations

import argparse
import collections
import platform
import sys
import threading
import time

import numpy as np

from neurosdk.brainbit_2_sensor import BrainBit2Sensor
from neurosdk.cmn_types import SensorCommand, SensorFamily, SensorFeature
from neurosdk.scanner import Scanner

from brainviz import amp_raw
from brainviz.source import make_channels, packets_to_array


def banner(text: str) -> None:
    print(f"\n=== {text} " + "=" * max(0, 70 - len(text)))


def attempt(label, fn, default=None):
    try:
        return fn()
    except Exception as exc:
        print(f"  {label}: <unavailable: {exc}>")
        return default


def scan(args):
    banner("Scan")
    names = ["LEBrainBitFlex", "LEBrainBitPro", "LEBrainBit2"]
    if args.scan_all:
        names += ["LEBrainBit", "LEBrainBitBlack", "LEHeadband", "LEHeadPhones2", "LECallibri", "LENeuroEEG"]
    families = [getattr(SensorFamily, n) for n in names]
    print("  families:", ", ".join(names))
    found: dict = {}
    got = threading.Event()

    def on_found(_scanner, sensors):
        for s in sensors:
            if s.Address not in found:
                print(f"  found: family={s.SensFamily.name} model={s.SensModel} name={s.Name!r} "
                      f"addr={s.Address} serial={s.SerialNumber} rssi={s.RSSI} pairing={s.PairingRequired}")
            found[s.Address] = s
        if sensors:
            got.set()

    scanner = Scanner(families)
    scanner.sensorsChanged = on_found
    scanner.start()
    print(f"  scanning up to {args.scan_seconds:g} s ...")
    if got.wait(args.scan_seconds):
        time.sleep(1.5)  # collect neighbours too
    scanner.stop()
    return scanner, list(scanner.sensors()) or list(found.values())


def connect(scanner, info, timeout=40.0):
    banner("Connect")
    print(f"  connecting to {info.Name} ({info.Address}); create_sensor blocks, so it runs on a thread ...")
    box: dict = {}

    def work():
        try:
            box["sensor"] = scanner.create_sensor(info)
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout)
    if "error" in box:
        raise SystemExit(f"  connect failed: {box['error']}")
    if "sensor" not in box:
        raise SystemExit(f"  connect did not finish within {timeout:g} s")
    return box["sensor"]


def describe(sensor):
    banner("Device")
    print("  python class :", type(sensor).__name__)
    print("  name         :", attempt("name", lambda: sensor.name))
    print("  address      :", attempt("address", lambda: sensor.address))
    print("  serial       :", attempt("serial", lambda: sensor.serial_number))
    print("  state        :", attempt("state", lambda: sensor.state))
    v = attempt("version", lambda: sensor.version)
    if v:
        print(f"  firmware     : {v.FwMajor}.{v.FwMinor}.{v.FwPatch}   hardware: {v.HwMajor}.{v.HwMinor}.{v.HwPatch}")
    print("  battery %    :", attempt("battery", lambda: sensor.batt_power))
    print("  sample rate  :", attempt("sampling_frequency", lambda: sensor.sampling_frequency))
    print("  channels     :", attempt("channels_count", lambda: sensor.channels_count))
    print("  amp mode     :", attempt("amp_mode", lambda: sensor.amp_mode))
    print("  supports     : signal=%s resist=%s signal+resist=%s" % tuple(
        attempt("feature", lambda f=f: sensor.is_supported_feature(f))
        for f in (SensorFeature.Signal, SensorFeature.Resist, SensorFeature.SignalWithResist)))
    cmds = {c.name: attempt(c.name, lambda c=c: sensor.is_supported_command(c))
            for c in (SensorCommand.StartSignal, SensorCommand.StopSignal, SensorCommand.StartResist,
                      SensorCommand.StopResist, SensorCommand.Idle, SensorCommand.PowerDown,
                      SensorCommand.StartSignalAndResist)}
    print("  commands     :", cmds)
    raw = attempt("raw amp params", lambda: amp_raw.snapshot(amp_raw.read_raw(sensor), sensor.channels_count))
    print("  raw amp bytes:", raw, "  (mode: 0=Short 1=Normal; gain: 2=Gain3)")
    banner("Channels (supported_channels)")
    infos = sensor.supported_channels
    for i in sorted(infos, key=lambda c: c.Num):
        print(f"  Num={i.Num}  Id={i.Id.name:<12} Type={i.ChType.name:<22} Name={i.Name!r}")
    channels = make_channels(infos)
    print("  labels used  :", [c.label for c in channels])
    return channels


def capture(sensor, channels, seconds, title):
    banner(title)
    nums = [c.num for c in channels]
    labels = [c.label for c in channels]
    chunks: list = []
    arrivals: list = []
    errors: list = []

    def on_signal(_sensor, data):
        try:
            chunks.append(packets_to_array(data, nums))
            arrivals.append((time.monotonic(), len(data)))
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    sensor.signalDataReceived = on_signal
    sensor.exec_command(SensorCommand.StartSignal)
    time.sleep(seconds)
    sensor.exec_command(SensorCommand.StopSignal)
    sensor.signalDataReceived = None
    time.sleep(0.3)

    print("  amp mode now :", attempt("amp_mode", lambda: sensor.amp_mode))
    if errors:
        print("  packet errors:", errors[:3])
    if not chunks:
        print("  NO SAMPLES RECEIVED.")
        return None
    pack = np.concatenate([c[0] for c in chunks])
    marker = np.concatenate([c[1] for c in chunks])
    uv = np.concatenate([c[2] for c in chunks])
    span = arrivals[-1][0] - arrivals[0][0]
    total = sum(n for _, n in arrivals)
    hz = (total - arrivals[0][1]) / span if span > 0 else float("nan")
    print(f"  callbacks    : {len(chunks)}, samples: {len(uv)}, samples/callback: "
          f"{collections.Counter(n for _, n in arrivals).most_common(3)}")
    print(f"  measured rate: {hz:.1f} Hz   (expected ~250)")
    steps = collections.Counter(np.diff(pack).tolist()).most_common(5)
    print(f"  PackNum      : first={pack[0]} last={pack[-1]}  step histogram (step, count): {steps}")
    print(f"  Marker values: {sorted(set(marker.tolist()))}")
    print("  first 5 samples (µV):")
    print("   " + " ".join(f"{l:>10}" for l in labels))
    for row in uv[:5]:
        print("   " + " ".join(f"{v:10.1f}" for v in row))
    print("  per channel (µV):  " + "  ".join(f"{l:>9}" for l in labels))
    print("    mean            " + "  ".join(f"{v:9.1f}" for v in uv.mean(0)))
    print("    std about mean  " + "  ".join(f"{v:9.2f}" for v in uv.std(0)))
    print("    min             " + "  ".join(f"{v:9.1f}" for v in uv.min(0)))
    print("    max             " + "  ".join(f"{v:9.1f}" for v in uv.max(0)))
    flat = np.where(uv.std(0) < 1e-6)[0]
    if len(flat) == len(labels):
        print("  VERDICT: every channel is constant. Data is flowing but carries no signal.")
        if np.allclose(uv, uv[0, 0]):
            print("  hint: the identical constant on every channel usually means floating/railed inputs "
                  "(headset not on the head, electrodes not touching). Put it on and re-run before suspecting "
                  "the amplifier configuration.")
    elif len(flat):
        print("  note: constant channels:", [labels[i] for i in flat])
    if np.abs(uv.mean(0)).max() > 5e5:
        print("  note: |mean| > 0.5 V, the unit scaling may be wrong.")
    return uv


def resist(sensor, channels, seconds):
    banner("Impedance (StartResist)")
    nums = [c.num for c in channels]
    last: list = []

    def on_resist(_sensor, data):
        last.append(np.asarray(data[-1].Samples, dtype=float))

    sensor.resistDataReceived = on_resist
    try:
        sensor.exec_command(SensorCommand.StartResist)
        time.sleep(seconds)
        sensor.exec_command(SensorCommand.StopResist)
    except Exception as exc:  # noqa: BLE001
        print("  resist failed:", exc)
    sensor.resistDataReceived = None
    if last:
        vals = last[-1][nums]
        print("  callbacks:", len(last))
        for c, v in zip(channels, vals):
            print(f"   {c.label:>8}: {v / 1e3:12.1f} kΩ")
    else:
        print("  no resistance packets received")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan-seconds", type=float, default=15.0)
    ap.add_argument("--capture-seconds", type=float, default=3.0)
    ap.add_argument("--gain", type=int, default=3, help="gain for phase B (default 3)")
    ap.add_argument("--scan-all", action="store_true", help="scan every sensor family (diagnostics)")
    ap.add_argument("--skip-config", action="store_true", help="skip phase B")
    ap.add_argument("--resist", action="store_true", help="also read impedance for 5 s")
    args = ap.parse_args()

    print(f"python {platform.python_version()} {platform.machine()} on {platform.platform()}")
    scanner, found = scan(args)
    usable = [s for s in found if s.SensFamily in (SensorFamily.LEBrainBitFlex, SensorFamily.LEBrainBitPro,
                                                   SensorFamily.LEBrainBit2)]
    if not usable:
        print("\nNo BrainBit2-family device found.")
        if found:
            print("Other devices seen (not supported by this probe):", [(s.Name, s.SensFamily.name) for s in found])
        print("Checklist: headset on and charged; BrainBit NeuroREC fully quit (BLE allows one client);\n"
              "terminal app has Bluetooth permission (System Settings > Privacy & Security > Bluetooth);\n"
              "try --scan-all to see what the SDK can see.")
        return 1
    usable.sort(key=lambda s: -s.RSSI)
    sensor = connect(scanner, usable[0])
    if not isinstance(sensor, BrainBit2Sensor):
        print(f"Connected object is {type(sensor).__name__}, not BrainBit2Sensor.")
        return 1
    try:
        channels = describe(sensor)
        capture(sensor, channels, args.capture_seconds, "Phase A: StartSignal as the device comes up")
        if not args.skip_config:
            banner(f"Configure: Idle, Normal mode, Gain{args.gain}")
            attempt("Idle", lambda: sensor.exec_command(SensorCommand.Idle))
            try:
                before, after = amp_raw.configure(sensor, args.gain, n_channels=sensor.channels_count)
                print("  before:", before)
                print("  after :", after)
            except Exception as exc:  # noqa: BLE001
                print("  configure failed:", exc)
            capture(sensor, channels, args.capture_seconds, f"Phase B: after amplifier config (Gain{args.gain})")
        if args.resist:
            resist(sensor, channels, 5.0)
    finally:
        banner("Cleanup")
        for step in (lambda: setattr(sensor, "signalDataReceived", None),
                     lambda: setattr(sensor, "resistDataReceived", None),
                     lambda: sensor.exec_command(SensorCommand.StopSignal),
                     lambda: sensor.disconnect()):
            try:
                step()
            except Exception:  # noqa: BLE001
                pass
        del sensor
        del scanner
        print("  done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
