# Single-person attention collection

This is a local pilot for repeatable EEG and behavioral data collection. It does not
measure a verified focused/unfocused state. No model, drowsiness detector, or live
attention percentage is included.

## Start a session

1. Close NeuroREC and other headset clients. Fit the electrodes and connect the reference
   and ground according to the headset instructions. Keep the arrangement consistent.
2. Run `.venv/bin/python run_visualizer.py --gradcpt`. For synthetic EEG use `--simulate` too.
   The launcher avoids an unrelated local `random.py` masking Python's standard library.
3. Enter the channel/electrode arrangement, reference placement and ground placement.
   Generic SDK labels are never interpreted as scalp positions. If a location is not
   known, record `unknown` explicitly rather than inventing it.
   Use **Record channel** to select all channels or one source channel. Typing a channel
   name into the arrangement note does not select the recording channel. For example,
   `--record-channel 2` preselects Ch2. The live plots still display all headset channels.
4. Click **Check electrode impedance…**, wait for readings, then **Done, start signal**.
   The 10-second preview starts again after signal resumes. Correct contact issues and
   click **Restart 10-second preview** if needed. There must be fresh streaming samples,
   a completed impedance reading, recorded placements, and no preview artifact flags.
5. Click **Start practice and record**. Press Space for city scenes and withhold for
   mountains. Continue trying normally throughout; do not deliberately create lapses.
   There are 150 practice transitions (2 minutes), 600 measured transitions (8 minutes),
   and an extra 800 ms scrambled tail after each phase to collect late responses.
6. The full-screen task ends automatically. Esc, loss of task focus, disconnect, stale
   packets or a write error ends the run and keeps partial files. Close/reopen the task
   to start another session. Stop before changing electrode positions.

For immediate raw recording, choose the channel and click **Start without checks**.
This skips impedance, the 10-second quality gate, and mandatory setup fields (blank details
are saved as `unknown`). It still requires a live signal and no other active recording.
The session and summary explicitly record `checks_skipped`; detected quality problems are
not cleared. Offline quality flags are calculated on the channels actually saved. You can
launch with `--skip-impedance` to prevent the automatic impedance pop-up.

Practice provides feedback on assigned keypresses and finalized earlier trials. Measured trials do not show correctness
feedback. Image transitions are continuous 800 ms fades. Each phase uses a shuffled exact
90/10 city/mountain split with no consecutive identical image. The seed and full sequence
are saved. Images are unmodified official TestMyBrain stimuli; provenance and licenses are
in `brainviz/assets/gradcpt/NOTICE.md`.

## What is saved

All new attention runs are inside Git-ignored `participant_data/<session-id>/` in the
working directory. `BRAINVIZ_DATA_DIR` can select a different local data directory:

- Raw EEG CSV and its JSON sidecar: existing column format and microvolt units are preserved.
- `*.packets.jsonl`: first sample index, batch count, Unix receipt time and monotonic receipt
  time. Receipts are captured before the data hub does display work.
- `session.json`: state, settings, placements, preflight, impedance, seed, stimulus provenance,
  file links, end reason and any processing error. `sequence.json` stores scheduled trials.
- `events.jsonl`: submitted trial onsets, rendered frames, raw presses, timing failures and end state.
- `trials.csv`: reference-scored responses, correctness, RT, VTC and behavioral label.
- `windows.csv`: 4-second windows at a 1-second stride, sample ranges, behavioral label,
  signal validity, training eligibility and explicit exclusion reasons.
- `summary.json`: behavioral accuracy/RT, usable EEG interval union, timing failures and window counts.

A global local-only `participant_data/setup.json` retains the last entered setup, not an
assumed verified montage. Reconfirm it whenever fitting the headset. Simulated files and
metadata are marked SIM; their windows are never eligible as real training EEG. The earlier
arithmetic CSV/events/metadata are unchanged. Neither collection nor export uploads anything.

## Label definitions and limits

Response scoring is a Python adaptation of the official TestMyBrain opacity-based
`assignTrial` / chronological code-4 correction. It can assign a press to the preceding
image and debounces presses within 50 ms. It does not simply label the currently visible
image as answered. The module is LGPL-3.0-or-later; license texts and change notices are bundled.

Completed measured runs use the published VTC approach: absolute within-run standardized
correct-city RTs (population standard deviation), linear interpolation, Gaussian smoothing
with 9-trial FWHM and a +/-10-trial support, and a within-run median split. Values below the
median get `stable_response`; above get `variable_response`; ties get `unknown`. These are
relative behavioral proxies, not ground-truth focus labels. A median split can divide an
otherwise strong run approximately in half.

Conservative pilot additions: at least 100 valid correct city responses, no extrapolated
edge labels, and gaps longer than 10 missing trials invalidate nearby smoothing support.
Practice, incomplete runs, insufficient behavior and degenerate RT/VTC remain `unknown`.
At least 75% of a window's full four seconds must have the same supported behavioral label;
otherwise it is unknown. These pilot policies are versioned, not independently validated.

Signal validity is independent of behavior. A window with gaps/resets, incomplete sampling,
invalid timing, nonfinite values, flat signals, provisional amplifier rail hits or large
filtered swings is excluded from training eligibility. All selected channels must pass; this version
does not silently drop or interpolate bad channels. Only explicitly selected recording
channels are saved and checked; an unselected noisy channel does not invalidate Ch2.
The raw data is always preserved.
Defaults: flat raw standard deviation <0.01 µV; saturation >=99% of provisional 2.4 V/gain;
1–45 Hz filtered peak-to-peak >500 µV (initial filter transient omitted); frame gap or initial
paint lateness >50 ms; stale EEG >1 s. Impedance thresholds remain provisional and displayed,
not a calibrated clinical quality score. Passing these rules does not prove clean neural EEG.

Thresholds are editable with a JSON file via `--quality-config file.json`; keys match
`QualityConfig` in `brainviz/attention_protocol.py`. The UI exposes the swing threshold.
Settings, version and policies are saved for reproducibility. No existing CSV is rewritten.
To rerun processing: `.venv/bin/python -P -m brainviz.attention_analysis SESSION_DIRECTORY`
from a clean checkout, or use Python with the project appended after standard-library paths
when unrelated interview scripts are present.

Timing uses a single host monotonic clock for events and packet receipts. Samples are mapped
by interpolation between batch-end receipts, with nominal-rate extrapolation only inside the
first batch. GUI paint submission is not a measured screen onset, BLE receipt is not the ADC
sample time, and device/display latency is unknown. Logs expose paint gaps, late/skipped
trials and alignment failures; hardware triggers and photodiode validation are not implemented.
VTC smoothing is offline and includes neighboring future behavior. For future model evaluation,
keep entire sessions together; do not randomly split overlapping windows into train and test.

## Manual sharing

Click **Export session…**, select a finished session folder, and choose a new ZIP filename.
The default destination is the Git-ignored `exports/` folder; ZIP files are also ignored.
The export contains only that session's known files, removes device serial and Bluetooth
address fields from JSON metadata, rejects symlinks/traversal, and leaves raw files unchanged.
EEG, responses and timestamps remain personal data. This is a local ZIP, not an upload.

## Verification and collection readiness

`QT_QPA_PLATFORM=offscreen .venv/bin/python -P scripts/verify_attention.py` exercises a short
real Qt/simulator run with synthetic responses. Add `--full` for the full 2+8-minute protocol.
The output includes frame submission timings and summary results and is explicitly marked
as simulated. Offscreen timing does not validate physical display latency.

With the visualizer/NeuroREC closed, `.venv/bin/python -P scripts/headset_preflight.py` runs an
impedance + 10-second raw signal-quality pilot and saves a local report without attention
labels. An actual task pilot requires you to wear the headset and perform the task; synthetic
responses are never attached to real EEG. Complete that pilot and review signal quality,
timing flags and labels before routine collection. A quality-failed pilot requires correcting
contact/setup and repeating the check.

Sources:
- Fortenbaugh et al., 2018, Methods 2.4.3: https://pmc.ncbi.nlm.nih.gov/articles/PMC5857436/
- Official task and stimuli: https://github.com/manybrainsproject/TestMyBrainCodeRepo
