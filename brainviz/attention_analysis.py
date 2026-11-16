"""Offline, reproducible behavioral proxies and approximate EEG-window alignment.

Run: python -m brainviz.attention_analysis participant_data/<session>
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .attention_protocol import QualityConfig
from .gradcpt_scoring import score_trials, SCORING_VERSION
from .quality import assess_quality
from .recorder import read_csv

ANALYSIS_VERSION = "vtc-pilot-1"
UNKNOWN = "unknown"


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compute_vtc(trials, config=None, complete=True):
    """Absolute z-score of correct city RTs, bounded interpolation, 9-trial FWHM.

    Minimum responses and maximum interpolation gap are conservative pilot policies,
    not parameters validated by the original publication. Median ties stay unknown.
    """
    config = config or QualityConfig()
    n = len(trials)
    values = np.full(n, np.nan)
    labels = np.full(n, UNKNOWN, dtype=object)
    good = np.array([i for i, t in enumerate(trials) if t["category"] == "city" and t["correct"]
                     and t["rt_s"] is not None and t["rt_s"] > 0 and not t.get("timing_bad", False)], dtype=int)
    if not complete or len(good) < config.min_valid_responses:
        return values, labels, "incomplete_or_insufficient_behavior"
    rt = np.array([trials[i]["rt_s"] for i in good])
    std = rt.std(ddof=0)
    if std < 1e-6:
        return values, labels, "degenerate_reaction_times"
    z = np.abs((rt - rt.mean()) / std)
    interpolated = np.interp(np.arange(n), good, z)
    supported = (np.arange(n) >= good[0]) & (np.arange(n) <= good[-1])
    for left, right in zip(good, good[1:]):
        if right - left - 1 > config.max_interpolation_gap_trials:
            supported[max(0, left - 10):min(n, right + 11)] = False
    sigma = 9 / np.sqrt(8 * np.log(2))
    smoothed = gaussian_filter1d(interpolated, sigma, radius=10, mode="nearest")
    values[supported] = smoothed[supported]
    if supported.sum() < config.min_valid_responses or np.ptp(values[supported]) < 1e-8:
        return np.full(n, np.nan), labels, "insufficient_or_degenerate_vtc"
    median = np.median(values[supported])
    labels[supported & (values < median)] = "stable_response"
    labels[supported & (values > median)] = "variable_response"
    return values, labels, "ok"


def window_label(trials, start, end):
    durations = {"stable_response": 0.0, "variable_response": 0.0}
    for t in trials:
        overlap = max(0.0, min(end, t["onset_s"] + t["duration_s"]) - max(start, t["onset_s"]))
        label = t.get("behavior_label", UNKNOWN)
        if label in durations and not t.get("timing_bad", False):
            durations[label] += overlap
    for label, duration in durations.items():
        if duration / (end - start) >= .75 - 1e-10:
            return label
    return UNKNOWN


def write_csv(path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path, doc):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sample_times(packets, count, fs):
    """Interpolate host receipt times at packet ends, not precise ADC timestamps."""
    ends = np.array([p["sample_start"] + p["sample_count"] - 1 for p in packets])
    times = np.array([p["host_monotonic_s"] for p in packets])
    if not len(ends) or ends[-1] != count - 1 or np.any(np.diff(ends) <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("Incomplete or non-monotonic packet timing")
    indices = np.arange(count)
    result = np.interp(indices, ends, times)
    first = indices < ends[0]
    result[first] = times[0] + (indices[first] - ends[0]) / fs
    return result


def process_session(directory):
    directory = Path(directory)
    meta = json.loads((directory / "session.json").read_text())
    config = QualityConfig(**meta["quality_config"])
    events = read_jsonl(directory / "events.jsonl")
    sequence = json.loads((directory / "sequence.json").read_text())
    paints = {e["trial_id"]: e for e in events if e["event"] == "trial_onset"}
    bad = set()
    for e in events:
        if e["event"] == "timing_failure":
            bad.update(e["trial_ids"])
    trials = []
    for phase in ("practice", "measured"):
        phase_rows = []
        for trial in sequence:
            if trial["phase"] not in (phase, phase + "_tail"):
                continue
            row = dict(trial)
            painted = paints.get(row["trial_id"])
            row["onset_s"] = painted["onset_s"] if painted else row["scheduled_onset_s"]
            row["timing_bad"] = painted is None or row["trial_id"] in bad
            phase_rows.append(row)
        presses = [e for e in events if e["event"] == "press" and e["phase"] in (phase, phase + "_tail")]
        trials.extend(score_trials(phase_rows, presses))
    measured = [t for t in trials if t["phase"] == "measured"]
    complete = meta["state"] == "complete"
    values, labels, label_status = compute_vtc(measured, config, complete)
    for trial in trials:
        trial.update(behavior_label=UNKNOWN, vtc=None)
    for trial, value, label in zip(measured, values, labels):
        trial.update(behavior_label=UNKNOWN if trial["timing_bad"] else label,
                     vtc=None if not np.isfinite(value) else float(value))
    header, data = read_csv(directory / meta["eeg_file"])
    packets = read_jsonl(directory / meta["packet_timing_file"])
    fs = meta["fs_hz"]
    alignment_error = None
    try:
        times = sample_times(packets, len(data), fs)
    except ValueError as exc:
        times = np.full(len(data), np.nan)
        alignment_error = str(exc)
    windows = []
    if measured:
        begin = measured[0]["scheduled_onset_s"]
        # Partial sessions have annotations only for the period actually recorded.
        end = min(measured[-1]["scheduled_onset_s"] + .8, meta["ended_monotonic_s"])
        for offset in range(max(0, int(np.floor(end - begin - 4 + 1e-6)) + 1)):
            start, stop = begin + offset, begin + offset + 4
            mask = (times >= start) & (times < stop)
            indices = np.flatnonzero(mask)
            reasons = []
            if alignment_error:
                reasons.append("invalid_packet_timing")
            if len(indices) < .98 * 4 * fs:
                reasons.append("missing_samples")
            quality = assess_quality(data[mask, 4:], fs, data[mask, 2], meta.get("gain"), config,
                                     meta.get("notch_hz", 60), channel_labels=header[4:])
            reasons.extend(quality["reasons"])
            overlap = [t for t in trials if start < t["onset_s"] + t["duration_s"] and stop > t["onset_s"]]
            if any(t["timing_bad"] for t in overlap):
                reasons.append("presentation_timing_failure")
            if len(indices) > 1 and np.max(np.diff(times[indices])) > .1:
                reasons.append("receipt_timing_gap")
            label = window_label(measured, start, stop) if complete else UNKNOWN
            valid = not reasons
            if label == UNKNOWN:
                reasons.append("unknown_behavior")
            if not complete:
                reasons.append("incomplete_session")
            if meta["simulated"]:
                reasons.append("simulated_data")
            windows.append({"start_monotonic_s": start, "end_monotonic_s": stop,
                            "sample_start": int(indices[0]) if len(indices) else "",
                            "sample_stop_exclusive": int(indices[-1] + 1) if len(indices) else "",
                            "behavior_label": label, "signal_valid": valid,
                            "training_eligible": valid and label != UNKNOWN and not meta["simulated"],
                            "exclusion_reasons": ";".join(dict.fromkeys(reasons))})
    rt = [t["rt_s"] for t in measured if t["category"] == "city" and t["correct"] and t["rt_s"] is not None]
    valid_intervals = sorted((w["start_monotonic_s"], w["end_monotonic_s"]) for w in windows if w["signal_valid"])
    duration = 0.0
    last_end = -float("inf")
    for start, end in valid_intervals:
        duration += max(0, end - max(start, last_end))
        last_end = max(last_end, end)
    measured_presented = [t for t in measured if t["trial_id"] in paints]
    summary = {"analysis_version": ANALYSIS_VERSION, "scoring_version": SCORING_VERSION,
               "checks_skipped": meta.get("checks_skipped", False),
               "recorded_channels": header[4:],
               "state": meta["state"], "simulated": meta["simulated"], "label_status": label_status,
               "accuracy": (sum(t["correct"] for t in measured_presented) / len(measured_presented)) if measured_presented else None,
               "mean_correct_city_rt_s": float(np.mean(rt)) if rt else None,
               "presented_measured_trials": len(measured_presented),
               "timing_problem_trials": sum(t["timing_bad"] for t in measured_presented),
               "usable_eeg_duration_s": round(duration, 3), "samples": len(data),
               "window_counts": {name: sum(w["behavior_label"] == name for w in windows)
                                 for name in ("stable_response", "variable_response", UNKNOWN)},
               "training_eligible_windows": sum(w["training_eligible"] for w in windows),
               "alignment_error": alignment_error,
               "interpretation": "Relative behavioral consistency; not verified focus. Median split is within-run. "
                                 "Timing is approximate host receipt/paint timing, not hardware ADC/display timing.",
               "processing": {"window_s": 4, "stride_s": 1, "label_agreement": .75,
                              "vtc_fwhm_trials": 9, "vtc_radius_trials": 10, "zscore_ddof": 0}}
    write_csv(directory / "trials.csv", trials, ["trial_id", "phase", "category", "image", "onset_s", "duration_s",
              "response_s", "rt_s", "response_code", "correct", "outcome", "timing_bad", "vtc", "behavior_label"])
    write_csv(directory / "windows.csv", windows, ["start_monotonic_s", "end_monotonic_s", "sample_start",
              "sample_stop_exclusive", "behavior_label", "signal_valid", "training_eligible", "exclusion_reasons"])
    write_json(directory / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    print(json.dumps(process_session(parser.parse_args().session), indent=2))
