"""Queue dispatch rules from lerobot-piper's piper_client.py, in SI units.

Reference: piper_scripts/piper_inference/piper_client.py, revision
8fff0fde7c79f23a93d845d1a50e985de01f8b8a, inspected file SHA256
04bf989456259e62b621406eb4563747394bf6df2c60331fb2effb8dfa8f7763.
Lines 133-154 define limits; 218-246 clip then limit relative to observation;
926-994 dispatch one queued target per tick and repeat the last target during
a fixed settling interval. Model chunk preprocessing/smoothing is separate.

The reference hardware policy retains fault, freshness and target checks and
uses device interpolation at the configured speed percentage. The separate
projected policy retains software slew checks. These queue rules do not test
arrival or infer task/contact success.
"""
from __future__ import annotations

import numpy as np


SOURCE = {
    "repository": "lerobot-piper",
    "revision": "8fff0fde7c79f23a93d845d1a50e985de01f8b8a",
    "file": "piper_scripts/piper_inference/piper_client.py",
    "sha256": "04bf989456259e62b621406eb4563747394bf6df2c60331fb2effb8dfa8f7763",
    "scope": "Queue dispatch only; no model preprocessing or automatic smoothing",
}
LOWER = np.r_[np.deg2rad([-154, 0, -175, -102, -75, -170]), 0.0]
UPPER = np.r_[np.deg2rad([154, 195, 0, 102, 75, 170]), 0.07]
MAX_DELTA = np.r_[np.full(6, np.deg2rad(45.0)), 0.03]
SETTLE_TIME_S = 0.5


def reference_limits(hardware_lower, hardware_upper):
    """Intersect the reference target limits with configured hardware limits."""
    lower = np.maximum(LOWER, hardware_lower)
    upper = np.minimum(UPPER, hardware_upper)
    if np.any(lower >= upper):
        raise ValueError("Reference and hardware target ranges must overlap")
    return lower, upper


def reference_target(action, measured, lower, upper):
    """Original order: target clip, then delta relative to fresh measurement."""
    clipped = np.clip(action, lower, upper)
    delta = np.clip(clipped - measured, -MAX_DELTA, MAX_DELTA)
    return np.asarray(measured) + delta
