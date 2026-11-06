"""Dataset-derived PAP reset state, independent of model and transport code."""
from __future__ import annotations

import numpy as np


def load_policy_initial(path, maximum_width_m):
    """Read only the first recorded observation; never load expert action targets."""
    with np.load(path, allow_pickle=False) as data:
        raw = np.asarray(data["state_raw"][0], dtype=float)
    if raw.shape != (7,) or not np.isfinite(raw).all():
        raise ValueError("Expected a finite seven-component initial dataset state")
    initial = np.r_[np.deg2rad(raw[:6]), raw[6] * .01]
    if not 0 <= initial[6] <= maximum_width_m:
        raise ValueError("Initial observed gripper width exceeds the configured travel")
    return dict(initial=initial, times=np.array([0.]), commands=np.empty((0, 7)),
                metadata=dict(source="first recorded observation only", source_angle_unit="degree",
                    source_width_unit="cm", width_unit_status="inferred from dataset and grasp geometry",
                    expert_action_targets_loaded=False, initial_state_only=True))

