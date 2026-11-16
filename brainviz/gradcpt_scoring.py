# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2024 The Many Brains Project, Inc. and McLean Hospital.
# Python adaptation of assignTrial/report, GradCPT_Main.v1.Apr24.html.
# Required copyright, change notice, and license texts: assets/gradcpt/NOTICE.md.
"""Offline TMB opacity-based assignment, including chronological code-4 correction."""
from __future__ import annotations

import bisect

SCORING_VERSION = "tmb-apr24-python-1"


def assign_trial(op: float, previous: bool, current: bool, prev_go: int, go: int):
    """Return (relative trial -1/0, reference response code), or None."""
    if previous and current or previous and op < .4 or current and op > .8:
        return None
    if not previous and op < .4:
        return -1, 1
    if not current and op > .8:
        return 0, 2
    if .4 <= op <= .8:
        if previous and not current:
            return 0, 3
        if current and not previous and prev_go == 1:
            return -1, 4
        if not previous and not current:
            if go != 1 and prev_go == 1:
                return -1, 5
            if go != 0 and prev_go != 1:
                return 0, 6
            if go == 0 and prev_go == 2:
                return 0, 10
            if go == prev_go:
                return (0, 7) if op > .6 else (-1, 8)
    return None


def score_trials(trials: list[dict], presses: list[dict]) -> list[dict]:
    """Score one phase, with an optional final scrambled tail for late responses.

    Presses optionally provide trial_id/opacity captured at the last rendered frame.
    Otherwise determine the current transition from onset timestamps (test/offline use).
    """
    rows = [dict(t, response_s=None, response_code=None) for t in trials]
    if not rows:
        return rows
    onsets = [t["onset_s"] for t in rows]
    ids = {t["trial_id"]: i for i, t in enumerate(rows)}
    go = [1 if t["category"] == "city" else 0 if t["category"] == "mountain" else 2 for t in rows]
    last_press = -float("inf")
    for press in sorted(presses, key=lambda p: p["time_s"]):
        ts = press["time_s"]
        if ts - last_press < .05:
            continue
        last_press = ts
        n = ids.get(press.get("trial_id"), bisect.bisect_right(onsets, ts) - 1)
        if n < 0 or ts >= onsets[-1] + rows[-1]["duration_s"]:
            continue
        op = press.get("opacity", (ts - onsets[n]) / rows[n]["duration_s"])
        previous = n > 0 and rows[n - 1]["response_s"] is not None
        current = rows[n]["response_s"] is not None
        assignment = assign_trial(op, previous, current, go[n - 1] if n else 2, go[n])
        if assignment is None:
            continue
        relative, code = assignment
        target = n + relative
        if target < 0:
            continue
        if code == 4:
            # An earlier press belonged to the previous image; swap chronological assignment.
            rows[target]["response_s"] = rows[n]["response_s"]
            rows[n]["response_s"] = ts
        else:
            rows[target]["response_s"] = ts
        rows[target]["response_code"] = code
    for row in rows:
        response = row["response_s"]
        row["rt_s"] = None if response is None else response - row["onset_s"]
        row["correct"] = (response is not None) == (row["category"] == "city")
        row["outcome"] = ("hit" if response is not None else "omission") if row["category"] == "city" else (
            "commission" if response is not None else "correct_withhold")
    return [r for r in rows if r["category"] != "scrambled"]
