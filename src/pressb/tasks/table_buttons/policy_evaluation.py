"""CPU-only evaluation helpers for physical table-button policy rollouts.

Scene and outcome checks are task-owned. Historical HTTP clients are available
lazily under their original names; their implementation lives in pressb.compat.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from pressb.control.joint_actions import JointActionPostprocessor
from .collection import validate_raw_arrays
from .task import COLORS, task_instruction
from .collision_policy import (
    FINGER_PAIR, DUAL_COLLISION_POLICY, STRICT_COLLISION_POLICY,
    collision_settings, is_allowed_finger_closure, strict_outcome,
)
from .policy_contract import (
    SCALE, POLICY_TO_SIM, InvalidPolicyAction, joint7, to_policy_units, to_sim_units,
)


__all__ = [
    "SCALE", "POLICY_TO_SIM", "FINGER_PAIR", "DUAL_COLLISION_POLICY", "STRICT_COLLISION_POLICY",
    "InvalidPolicyAction", "canonical_sha", "file_sha", "collision_settings",
    "is_allowed_finger_closure", "joint7", "to_policy_units", "to_sim_units",
    "command_limits", "JointPolicyCommands", "png_base64", "HTTPJointPolicy", "strict_outcome",
    "load_cell", "episode_config", "validate_execution_arrays", "summarize",
]


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_limits(dof_limits):
    """Derive full-width bounds from the actual eight articulation DOFs."""
    bounds = np.asarray(dof_limits, dtype=float)
    if bounds.shape != (8, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Expected finite increasing physical limits for joint1..8")
    lower_width = max(0., 2 * bounds[6, 0], -2 * bounds[7, 1])
    upper_width = min(2 * bounds[6, 1], -2 * bounds[7, 0])
    if upper_width <= lower_width:
        raise ValueError("Physical finger limits do not permit symmetric gripper motion")
    return np.r_[bounds[:6, 0], lower_width], np.r_[bounds[:6, 1], upper_width]


class JointPolicyCommands:
    """Explicit units -> physical limit clamp -> optional arm-only smoothing."""
    def __init__(self, lower, upper, initial, *, smoothing="none", alpha=.8):
        self.lower, self.upper = joint7(lower), joint7(upper)
        if np.any(self.lower >= self.upper):
            raise ValueError("Invalid joint7 physical limits")
        self.postprocessor = JointActionPostprocessor(smoothing, alpha)
        self.postprocessor.reset(np.clip(joint7(initial), self.lower, self.upper))
        self.clipped_by_dimension = np.zeros(7, dtype=int)
        self.clipped_actions = 0
        self.total_actions = 0

    def apply(self, raw):
        requested = joint7(raw)
        converted = to_sim_units(requested)
        clipped = np.clip(converted, self.lower, self.upper)
        mask = clipped != converted
        applied = self.postprocessor.apply(clipped)
        self.clipped_by_dimension += mask.astype(int)
        self.clipped_actions += int(mask.any())
        self.total_actions += 1
        return applied, dict(raw_command_deg_cm=requested.tolist(), converted_rad_m=converted.tolist(),
            clipped_rad_m=clipped.tolist(), applied_rad_m=applied.tolist(),
            clipped_dimensions=np.flatnonzero(mask).tolist())

    def summary(self):
        return dict(total_actions=self.total_actions, clipped_actions=self.clipped_actions,
            clipped_by_dimension=self.clipped_by_dimension.tolist(), lower_rad_m=self.lower.tolist(),
            upper_rad_m=self.upper.tolist(), smoothing=self.postprocessor.summary(),
            clipping_rule="physical joint limits only; any clamp is recorded, not automatically a failure")


def load_cell(plan_path, cell_id):
    """Load immutable per-cell plans; single/multitask cells share layout files."""
    plan = json.loads(Path(plan_path).read_text())
    if plan.get("schema") != "pressb.table_buttons.policy_eval_plan.v1":
        raise ValueError("Unexpected policy evaluation plan schema")
    selected = [cell for cell in plan["cells"] if cell["cell_id"] == cell_id]
    if len(selected) != 1:
        raise ValueError("Evaluation plan must contain exactly one matching cell")
    cell = selected[0]
    defaults = plan["defaults"]
    collision_settings(defaults.get('collision_policy'))
    if defaults["fps"] != 30 or abs(defaults["physics_dt"] - 1 / 120) > 1e-12 or defaults["execute_steps"] != 7:
        raise ValueError("Evaluation requires seven actions per request, 30 Hz control, 120 Hz physics")
    if cell["instruction"] != task_instruction(cell["target_color"]) or cell["spacing_profile"] not in ("near", "medium", "far"):
        raise ValueError("Cell task identity is inconsistent")
    ids = [row["episode_index"] for row in cell["episodes"]]
    if ids != list(range(len(ids))) or len({row["seed"] for row in cell["episodes"]}) != len(ids):
        raise ValueError("Episode indices/seeds must be distinct and ordered")
    for row in cell["episodes"]:
        if type(row["seed"]) is not int or row["seed"] < 10000 or type(row["policy_seed"]) is not int:
            raise ValueError("Held-out layouts require new seeds >=10000 and explicit paired policy seeds")
    return plan, cell


def episode_config(plan, cell, row):
    """Verify scene config against common collection settings and paired layout."""
    for name in ('layout', 'effective_config', 'preflight'):
        if name + '_sha256' in row and file_sha(row[name + '_file']) != row[name + '_sha256']:
            raise ValueError(f"Frozen {name} file changed after evaluation planning")
    layout = json.loads(Path(row["layout_file"]).read_text())
    actual = json.loads(Path(row["effective_config_file"]).read_text())
    expected = deepcopy(plan["config"])
    expected["position_sampling"]["spacing_profile"] = cell["spacing_profile"]
    expected["task_color"] = cell["target_color"]
    expected["task_instruction"] = cell["instruction"]
    for color in COLORS:
        expected["buttons"][color]["center_xy"] = layout["positions"][color]
    expected["press_approaches"] = layout["approaches"]
    expected["sampling_domain"] = layout["domain"]
    if actual != expected:
        raise ValueError("Effective scene differs from frozen collector configuration and paired layout")
    return actual, layout


def validate_execution_arrays(frames, physics, *, fps=30, physics_dt=1 / 120):
    if len(frames["sim_time"]) == 1:
        if (np.asarray(frames["state"]).shape != (1, 7) or np.asarray(frames["action"]).shape != (1, 7)
                or np.asarray(physics["q_actual"]).shape != (1, 8)
                or np.asarray(physics["q_target"]).shape != (1, 8)
                or frames["sim_time"][0] != 0 or physics["time_s"].tolist() != [0.]):
            raise ValueError("Invalid zero-action episode arrays")
        for array in (*frames.values(), *physics.values()):
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite zero-action episode arrays")
        return
    validate_raw_arrays(frames, physics, fps=fps, physics_dt=physics_dt)


def summarize(output, cell):
    reports = []
    for row in cell["episodes"]:
        path = Path(output) / "episodes" / f"episode_{row['episode_index']:06d}" / "report.json"
        if path.exists():
            report = json.loads(path.read_text())
            if report.get("evaluation_status") != "complete" or report["cell_id"] != cell["cell_id"] or report["seed"] != row["seed"]:
                raise ValueError(f"Existing episode identity/completion mismatch: {path}")
            reports.append(report)
    counts = {}
    for report in reports:
        counts[report["stop_reason"]] = counts.get(report["stop_reason"], 0) + 1
    successes = sum(report["success"] for report in reports)
    strict_successes = sum(report.get('strict_collector_success', report['success']) for report in reports)
    return dict(cell_id=cell["cell_id"], model_id=cell["model_id"], spacing_profile=cell["spacing_profile"],
                target_color=cell["target_color"], planned=len(cell["episodes"]), completed=len(reports),
                successes=successes, failures=len(reports) - successes,
                success_rate=successes / len(reports) if reports else None,
                task_successes=successes, task_success_rate=successes / len(reports) if reports else None,
                strict_collector_successes=strict_successes,
                strict_collector_success_rate=strict_successes / len(reports) if reports else None,
                stop_reasons=counts, complete=len(reports) == len(cell["episodes"]))


def __getattr__(name):
    if name in {"HTTPJointPolicy", "png_base64"}:
        from pressb.compat import table_buttons_policy
        return getattr(table_buttons_policy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
