"""CPU-only diagnostics for contiguous joint7 trajectories; never filter rows.

Coordinates are six joint angles in radians followed by full gripper width in
metres. ``near_static`` means a small *per-control-interval* change, including
slow nonzero motion; it is not a physical declaration that the robot stopped.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np


NEAR_STATIC_JOINT_STEP_DEG = .05
NEAR_STATIC_GRIPPER_STEP_M = .0001


def _fps(value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("fps must be finite and positive") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("fps must be finite and positive")
    return value


def _series(values, name):
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[1] != 7 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain at least one finite seven-coordinate row")
    return result


def _motion(values, fps):
    delta = np.diff(values, axis=0)
    count = len(delta)
    exact = np.all(delta == 0., axis=1)
    near = (np.max(np.abs(delta[:, :6]), axis=1) < np.deg2rad(NEAR_STATIC_JOINT_STEP_DEG)) & (
        np.abs(delta[:, 6]) < NEAR_STATIC_GRIPPER_STEP_M)
    return dict(
        comparison_count=count,
        exact_duplicate_count=int(exact.sum()),
        exact_duplicate_fraction=float(exact.mean()) if count else 0.,
        near_static_interval_count=int(near.sum()),
        near_static_interval_fraction=float(near.mean()) if count else 0.,
        slow_nonzero_interval_count=int((near & ~exact).sum()),
        max_joint_step_deg=float(np.rad2deg(np.max(np.abs(delta[:, :6])))) if count else 0.,
        max_gripper_step_m=float(np.max(np.abs(delta[:, 6]))) if count else 0.,
        near_static_thresholds=dict(
            joint_step_deg=NEAR_STATIC_JOINT_STEP_DEG,
            full_gripper_width_step_m=NEAR_STATIC_GRIPPER_STEP_M,
            joint_rate_deg_s=NEAR_STATIC_JOINT_STEP_DEG*fps,
            full_gripper_width_rate_m_s=NEAR_STATIC_GRIPPER_STEP_M*fps,
            comparison="strictly less than both thresholds on every coordinate",
            meaning="Includes exact duplicates and slow nonzero motion; not a physical stop classifier",
        ),
    )


def plan_motion_metrics(plan, initial_state=None):
    """Measure dispatched targets without interpreting repeats as actual stops.

    N targets execute for N/fps seconds and generate N+1 boundary observations.
    Without ``initial_state``, only N-1 adjacent target pairs are comparable.
    With it, the first comparison is measured initial state to first target;
    the other N-1 comparisons remain target-to-target, not measured motion.
    Planned holds are counted separately, even if command noise changed them.
    """
    actions = _series(plan["actions"], "actions")
    fps = _fps(plan["fps"])
    phases = plan.get("phases")
    if phases is not None and (len(phases) != len(actions) or not all(isinstance(p, str) for p in phases)):
        raise ValueError("phases must contain one string for each action")
    values = actions
    if initial_state is not None:
        initial = np.asarray(initial_state, dtype=float)
        if initial.shape != (7,) or not np.isfinite(initial).all():
            raise ValueError("initial_state must be a finite seven-coordinate row")
        values = np.concatenate([initial[None], actions], axis=0)
    holds = Counter()
    for waypoint in plan.get("waypoints", []):
        seconds = float(waypoint.get("hold_seconds", 0.))
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("waypoint hold_seconds must be finite and nonnegative")
        phase = waypoint.get("phase", "unspecified")
        if not isinstance(phase, str):
            raise ValueError("waypoint phase must be a string")
        holds[phase] += round(seconds*fps)
    hold_frames = sum(holds.values())
    return dict(
        schema_version=1, kind="planned_joint7_targets", fps=fps,
        action_count=len(actions), expected_recorded_frame_count=len(actions)+1,
        duration_s=len(actions)/fps,
        comparison_reference="initial_state_then_targets" if initial_state is not None else "adjacent_targets_only",
        phase_frames=dict(Counter(phases)) if phases is not None else {},
        planned_hold_frames=hold_frames, planned_hold_seconds=hold_frames/fps,
        planned_hold_fraction=hold_frames/len(actions), planned_hold_frames_by_phase=dict(holds),
        max_gripper_width_m=float(actions[:, 6].max()),
        min_gripper_width_m=float(actions[:, 6].min()),
        row_filtering=False, retimed=False,
        **_motion(values, fps),
    )


def measured_motion_metrics(states, fps):
    """Measure N boundary states over N-1 real control intervals at ``fps``."""
    values = _series(states, "states")
    fps = _fps(fps)
    return dict(
        schema_version=1, kind="measured_joint7_states", fps=fps,
        sample_count=len(values), interval_count=len(values)-1,
        duration_s=(len(values)-1)/fps,
        comparison_reference="adjacent_measured_states",
        max_gripper_width_m=float(values[:, 6].max()),
        min_gripper_width_m=float(values[:, 6].min()),
        row_filtering=False, retimed=False,
        **_motion(values, fps),
    )
