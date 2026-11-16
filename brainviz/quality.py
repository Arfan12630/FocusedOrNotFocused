"""Conservative artifact flags, not a physiological EEG validation or diagnosis."""
import numpy as np
from .attention_protocol import QualityConfig
from .dsp import StreamFilter


def assess_quality(raw, fs, pack=None, gain=3, config=None, notch=60.0, channel_labels=None):
    config = config or QualityConfig()
    raw = np.asarray(raw, dtype=float)
    flags = []
    if raw.ndim != 2 or len(raw) < fs:
        return {"valid": False, "reasons": ["insufficient_samples"], "channels": []}
    if pack is not None and np.any(np.diff(np.asarray(pack, dtype=np.int64)) != 1):
        flags.append("packet_gap_or_reset")
    channels = []
    rail_uv = 2.4e6 / gain if gain else None  # observed Flex8 full-scale; provisional
    for i in range(raw.shape[1]):
        x = raw[:, i]
        reasons = []
        if not np.isfinite(x).all():
            reasons.append("nonfinite")
        else:
            if np.std(x) < config.flat_std_uv:
                reasons.append("constant_signal")
            if rail_uv and np.any(np.abs(x) >= rail_uv * config.rail_fraction):
                reasons.append("amplifier_saturation")
            filt = StreamFilter(fs, 1, notch=notch).process(x[:, None])[:, 0]
            settled = filt[min(int(fs), len(filt) // 2):]
            if np.ptp(settled) > config.swing_peak_to_peak_uv:
                reasons.append("large_filtered_swing")
        label = channel_labels[i] if channel_labels is not None else f"channel_{i + 1}"
        channels.append({"channel_index": i, "label": label, "reasons": reasons})
        flags.extend(f"{label}:{r}" for r in reasons)
    return {"valid": not flags, "reasons": flags, "channels": channels}
