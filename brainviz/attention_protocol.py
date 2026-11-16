"""Reproducible protocol and versioned provisional quality settings."""
from dataclasses import asdict, dataclass
import math
import numpy as np

PROTOCOL_VERSION = "gradcpt-pilot-1"
TRIAL_S = .8


@dataclass(frozen=True)
class QualityConfig:
    version: str = "provisional-1"
    flat_std_uv: float = .01
    swing_peak_to_peak_uv: float = 500.0
    rail_fraction: float = .99
    frame_gap_s: float = .05
    min_valid_responses: int = 100
    max_interpolation_gap_trials: int = 10
    preview_s: float = 10.0
    stale_s: float = 1.0

    def __post_init__(self):
        for name in ("flat_std_uv", "swing_peak_to_peak_uv", "frame_gap_s", "stale_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive finite value")
        if not 0 < self.rail_fraction <= 1 or not math.isfinite(self.preview_s) or self.preview_s < 1:
            raise ValueError("rail_fraction must be in (0, 1]; preview_s must be at least 1")
        if not isinstance(self.min_valid_responses, int) or not isinstance(self.max_interpolation_gap_trials, int) \
                or self.min_valid_responses < 3 or self.max_interpolation_gap_trials < 0:
            raise ValueError("Need at least 3 valid responses and a nonnegative interpolation limit")

    def to_dict(self):
        return asdict(self)


def make_sequence(seed: int, practice_trials: int = 150, measured_trials: int = 600) -> list[dict]:
    rng = np.random.default_rng(seed)
    result = []
    for phase, count in (("practice", practice_trials), ("measured", measured_trials)):
        categories = np.array(["mountain"] * round(count * .1) + ["city"] * (count - round(count * .1)))
        rng.shuffle(categories)
        previous = None
        for category in categories:
            choices = [f"{category}{i}.png" for i in range(1, 11) if f"{category}{i}.png" != previous]
            image = str(rng.choice(choices))
            result.append({"trial_id": len(result), "phase": phase, "category": str(category),
                           "image": image, "duration_s": TRIAL_S})
            previous = image
        result.append({"trial_id": len(result), "phase": phase + "_tail", "category": "scrambled",
                       "image": "scrambled.png", "duration_s": TRIAL_S})
    return result
