# FocusedOrNotFocused
Getting Brain Data

## BrainBit Flex8 visualizer

Real-time EEG viewer for the BrainBit Flex8 (BLE name "Flex Pro") on macOS, built on BrainBit's
`pyneurosdk2` and pyqtgraph.

- 8 scrolling traces, 1–45 Hz bandpass + mains notch (SciPy, streaming)
- delta / theta / alpha / beta / gamma band power (relative % or absolute µV²)
- spectrogram of the selected channel (or the all-channel mean)
- impedance check, and CSV recording with a metadata sidecar
- `--simulate` mode: fake EEG through the same code path, no headset needed

### Setup

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e ".[dev]"      # or: pyneurosdk2==1.0.15 numpy scipy pyqtgraph PyQt6 pytest
```

Python 3.13 is what this was developed and tested on. `pyneurosdk2` is pinned to exactly 1.0.15 because
`brainviz/amp_raw.py` uses private wrapper internals.

The SDK's bundled `libneurosdk2.dylib` is a universal (x86_64 + arm64) binary; it loads natively on Apple
Silicon, nothing extra to install.

### Before the first run with the headset

1. **Quit BrainBit NeuroREC** (and anything else connected to the headset). BLE normally allows one client.
2. **Bluetooth permission for your terminal app.** The SDK scans BLE itself, so the device never appears in
   System Settings > Bluetooth. Run from iTerm (it declares the Bluetooth usage string); macOS will ask once.
   If it was denied: System Settings > Privacy & Security > Bluetooth. If the process aborts instantly with
   no message, the host app lacks the permission.
3. Turn the headset on and charge it.

Then run the probe and read its output before the full app:

```sh
.venv/bin/python probe.py            # add --resist to read impedance, --scan-all to list every BrainBit family
```

It scans, connects, prints version, battery, channel table and raw amplifier bytes, then captures 3 s twice:
**phase A** with no configuration and **phase B** after writing Normal mode + Gain3. Compare them to see
whether the Flex8 needs amplifier parameters before data flows, and check the measured rate (~250 Hz),
the `PackNum` step histogram, and that amplitudes are plausible µV.

### Run

For attention collection, run:

```sh
.venv/bin/python run_visualizer.py --gradcpt
.venv/bin/python run_visualizer.py --gradcpt --simulate  # synthetic EEG only
```

This adds a local electrode/quality check, 2-minute practice and 8-minute city/mountain
task, followed by offline behavioral labels and EEG quality annotations. New sessions go
to Git-ignored `participant_data/`. **Export session…** creates a local ZIP with device
identifiers removed from JSON metadata. See [ATTENTION_PROTOCOL.md](ATTENTION_PROTOCOL.md)
for the collection procedure, file formats, timing limitations, thresholds and validation.
The safe launcher also avoids unrelated root-level scripts shadowing Python standard modules.

```sh
.venv/bin/python -m brainviz --simulate        # no headset
.venv/bin/python -m brainviz                   # real device
```

| flag | default | meaning |
|---|---|---|
| `--simulate` | off | fake EEG: occipital alpha, 1/f background, blinks, 60 Hz hum, mV-scale DC, fake impedance |
| `--gain N` | 3 | gain written to every channel (1, 2, 3, 4, 5, 6, 8, 12, 24). `none` leaves the device untouched |
| `--notch HZ` | 60 | mains notch; `50` for Europe, `0` disables |
| `--labels A,B,...` | from SDK | channel labels in channel order |
| `--skip-impedance` | off | don't open the impedance check after connecting |
| `--out DIR` | `recordings` | where CSVs go |
| `--seed N` | random | repeatable simulation |

**Flow:** connect → impedance check (signal pauses while it's open; **Done** resumes) → live view.
Toolbar: Record, Impedance…, Reconnect, trace scale, time window, channel for band power/spectrogram.
Bars are labelled; the selected channel's lane is highlighted. Impedance colours are always paired with a
glyph and a word, and the thresholds (good < 500 kΩ, fair < 2 MΩ) are **provisional**; they live in
`brainviz/ui.py`.

### Recordings

#### Arithmetic pilot

Click **Arithmetic…** (or launch with `--arithmetic --skip-impedance`). Once live EEG is
arriving, click **Start and record**: 3 practice questions followed by 20 measured addition/
subtraction questions. Choose answers using keys **1–4** or the buttons. Each question has a
12-second timeout. Practice gets correctness feedback; measured trials do not. Stop and save
keeps a partial session. EEG recording stops automatically at completion or interruption.

The usual raw EEG CSV/JSON are accompanied by `.events.jsonl`, containing question operands,
answer options, correct answers, practice/measured phase, onsets, responses, response times,
timeouts, and session end state. The JSON sidecar links the event file. Events use the same
host Unix clock as EEG packet arrival times; monotonic timestamps are also saved for events.
Alignment is approximate: GUI onset timestamps are not measured display onsets, and Bluetooth
receipt times are not ADC sample timestamps. No hardware triggers are emitted.

This is a short arithmetic pilot, not gradCPT or a validated attention calibration. Correctness
and response times are behavioral measurements, not automatic focused/unfocused labels.
Difficulty varies across questions; operands and operators are saved for later analysis.
Simulated sessions are marked `SIM` and must not be used as real participant EEG.

`recordings/YYYYmmdd-HHMMSS[-SIM].csv` plus a `.json` sidecar.

```
sample_index, host_time_unix, pack_num, marker, <one column per channel, µV>
```

- Data is **unfiltered** (DC and mains included), so any filtering is reproducible offline.
- `host_time_unix` is when the packet arrived at this computer, not the ADC sample time; samples that came in
  one packet share it. Use `sample_index / fs` for timing within a recording.
- The sidecar holds fs, channel table, gain / amplifier bytes, device info, filter settings, last impedance
  reading, an estimate of lost samples and a `simulated` flag.
- Recording runs on its own thread; a stalled UI does not lose samples. A dropped connection closes the file.

### Layout

```
probe.py                minimal hardware probe
brainviz/
  source.py             EEGSource interface, channel labelling, packets -> µV array
  sdk_source.py         real device: one worker thread owns all blocking SDK calls
  amp_raw.py            raw amplifier read/write (works around an SDK enum bug)
  sim_source.py         simulated Flex8
  ringbuffer.py         ring buffer, rate meter, PackNum loss estimate, DataHub
  dsp.py                streaming filter, Welch band power, spectrogram columns
  recorder.py           CSV writer thread
  ui.py                 pyqtgraph window + impedance dialog
scripts/smoke_ui.py     offscreen end-to-end UI test (QT_QPA_PLATFORM=offscreen)
tests/                  pytest suite (includes a fake-SDK test of the whole connect/impedance/reconnect flow)
```

### Tests

```sh
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest
QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/smoke_ui.py .smoke   # drives the real window, saves screenshots
```

### pyneurosdk2 1.0.15 quirks this project works around

- `BrainBit2ChannelMode.ChModeShort = 0,` has a trailing comma, so its value is the tuple `(0,)` and
  `BrainBit2ChannelMode(0)` raises. `sensor.amplifier_param` therefore fails whenever any channel is in Short
  mode. `amp_raw.py` reads/writes the native struct directly and changes only the mode and gain bytes.
  (A test documents the bug and will flag a fixed release.)
- The `GenCurrent` enum has fewer members than the README lists, so `Current` is carried through as a raw byte.
- `sensor.features` and `sensor.commands` allocate a one-element buffer for a multi-element result and look like
  buffer overruns; use `is_supported_feature` / `is_supported_command` instead.
- `BrainBit2Sensor` has no combined signal + resistance callback, so the impedance check pauses the signal.
- The native library creates an (empty) `logs/sdk_log.log` in whatever directory you run from. Harmless.
- `setup.py` lists the dylib with Windows-style backslashes in `package_data`; it still installs correctly
  on macOS (verified), but a future release could regress this.

### Verified on a real Flex8 (probe run, headset off the head)

- scan finds it as family `LEBrainBitFlex`, model 3, name `Flex8`; connects as `BrainBit2Sensor` (firmware 1.10.0)
- Bluetooth permission works from iTerm; the arm64 library runs natively
- 8 channels, 250 Hz (measured 249.2 and 249.7 Hz); `PackNum` steps by exactly 1 per sample with no gaps
- raw amplifier write works: Gain3 written and read back; mode, resist-use and current bytes preserved
- device defaults: Normal mode, Gain6, resist-use on, probe current 6 nA (`Current` = 1)
- `supported_channels` reports every electrode as `Type=Ref, Id=Unknown, Name='Any'`, so channels are labelled
  `Ch1`…`Ch8` unless you pass `--labels`
- `StartResist` runs and returns `inf` on every channel when no electrode touches skin
- with the headset off the head every channel reads one identical constant (0.4 V at Gain6, 0.8 V at Gain3,
  i.e. 2.4 V / gain). That looks like the amplifier at full scale on floating inputs, not a fault.

### Still to verify with the headset on the head

- real EEG amplitudes, and that traces are alive in the app (Gain3 default; `--gain` overrides)
- that `amp_mode` read-back is meaningful: it printed `PowerDown` even while samples were streaming
- sensible impedance thresholds (the 500 kΩ / 2 MΩ values are provisional)
- whether the floating-input rail clears once electrodes make contact (see above)

### Troubleshooting

| symptom | likely cause |
|---|---|
| "No BrainBit device found" | headset off/flat; NeuroREC still connected; terminal lacks Bluetooth permission |
| process exits immediately on scan | host terminal app has no Bluetooth permission |
| connected but "no samples after 5 s" | run `probe.py`; compare phase A and B |
| traces flat, huge DC | wrong amplifier mode; try `--gain 3`, check phase B |
| "Reconnect" greyed out | only enabled when disconnected or in error |
