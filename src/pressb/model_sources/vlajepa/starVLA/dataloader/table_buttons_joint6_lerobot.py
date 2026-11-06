"""Read unchanged joint7 LeRobot data with seven observations and six actions."""
from __future__ import annotations

import copy
import numpy as np

from .piper_lerobot import _canonical_sha, _require
from .table_buttons_lerobot import (
    TableButtonsLeRobotDataset, NORMALIZATION as STATE_NORMALIZATION,
    normalize_joint7, collate_fn,
)

FIXED_GRIPPER_FULL_WIDTH_M = 0.00019999999552965166
ACTION_NORMALIZATION = dict(
    kind="fixed_joint6_scale", version=1,
    feature_names=[f"joint_{index}.pos" for index in range(1, 7)],
    input_units=["degree"] * 6, scale=[180.] * 6, offset=[0.] * 6, clip=False,
    forward="normalized = degrees / scale", inverse="degrees = normalized * scale",
    fitted_statistics=False,
)
NORMALIZATION = dict(
    kind="separate_joint7_state_joint6_action", version=1,
    state=copy.deepcopy(STATE_NORMALIZATION), action=copy.deepcopy(ACTION_NORMALIZATION),
    fixed_gripper_full_width_m=FIXED_GRIPPER_FULL_WIDTH_M,
    nominal_fixed_gripper_full_width_m=.0002,
    fixed_gripper_source="constant source dataset action; measured state gripper remains observed",
)


def normalize_joint6(values):
    values = np.asarray(values)
    _require(values.shape[-1:] == (6,) and np.isfinite(values).all(), "Expected finite (...,6) degree actions")
    return values.astype(np.result_type(values.dtype, np.float32), copy=False) / np.float32(180.)


def denormalize_joint6(values):
    values = np.asarray(values)
    _require(values.shape[-1:] == (6,) and np.isfinite(values).all(), "Expected finite (...,6) normalized actions")
    return values.astype(np.result_type(values.dtype, np.float32), copy=False) * np.float32(180.)


class TableButtonsJoint6LeRobotDataset(TableButtonsLeRobotDataset):
    """Same split, RGB preprocessing, states and timing; omit gripper targets.

    The parent verifies the original seven-channel files and source provenance.
    Only the in-memory action array is sliced; the immutable dataset is intact.
    """
    state_dim, action_dim = 7, 6
    normalize = staticmethod(normalize_joint6)
    denormalize = staticmethod(denormalize_joint6)
    normalize_state = staticmethod(normalize_joint7)
    normalize_action = staticmethod(normalize_joint6)
    denormalize_action = staticmethod(denormalize_joint6)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        expected = np.float32(FIXED_GRIPPER_FULL_WIDTH_M * 100.) / np.float32(8.)
        _require(np.all(self._actions[:, 6] == expected),
                 "Joint6 fixed-gripper policy requires the same constant 0.2 mm source action")
        self._actions = self._actions[:, :6].copy()
        self._actions.flags.writeable = False
        self.normalization = copy.deepcopy(NORMALIZATION)
        self.fingerprint = dict(self.fingerprint, adapter=type(self).__name__,
            state_dim=7, action_dim=6, normalization=copy.deepcopy(self.normalization))
        self.fingerprint_sha256 = _canonical_sha(self.fingerprint)
