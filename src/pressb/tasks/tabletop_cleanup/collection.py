"""Task hooks for collecting diverse physical cleanup demonstrations.

The collector owns simulation, cameras, action perturbations and dataset output.
These hooks own the exact-two-object sampling, settled-pose plan and acceptance
criteria. Object transforms are inputs to planning, never replayed as actions.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .configuration import sample_objects
from .geometry import KINDS
from .planning import plan_cleanup
from .task import INSTRUCTION

ROOT = Path(__file__).resolve().parents[4]
PAIRS = tuple(combinations(KINDS, 2))


def _merge(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = deepcopy(value)


def collection_config(base_config=None):
    """Raw task config; ranges bound complete footprints, not object centres."""
    config = json.loads((ROOT / "configs/rl_nodes/tabletop_cleanup_real_layout.json").read_text())["backend"]["config"]
    config.update(horizon=1, terminate_on_success=False, max_episode_seconds=120.)
    config["initial_pose"]=dict(mode="workspace_center",height_above_table_m=.16,
                                radial_tilt_deg=15.,tool_yaw_deg=180.,gripper_width_m=.1)
    config["randomization"].update(enabled=True, fixed_subset=None,
        pool=list(KINDS), x_range_m=[.24, .46], y_range_m=[-.22, .13],
        yaw_range_deg=[-30., 30.], clearance_m=.030, max_attempts=4096)
    if base_config is not None:
        updates = base_config.get("backend", {}).get("config", base_config)
        _merge(config, updates)
    return config


def sample_episode(normalized_config, seed, episode_index, env_id=0):
    """Balance all three pairs by successful dataset slot, not retry count.

    Failed attempts must retain episode_index and change seed. Thus 500 accepted
    episodes contain pair counts 167, 167 and 166 even with unequal failures.
    Layout proposals use the task's existing complete-footprint validation.
    """
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError("episode_index must be a nonnegative integer")
    config = deepcopy(normalized_config)
    pair = PAIRS[episode_index % len(PAIRS)]
    if any(kind not in config["randomization"]["pool"] for kind in pair):
        raise ValueError("Balanced cleanup collection requires all three object kinds")
    config["randomization"]["fixed_subset"] = list(pair)
    layout = sample_objects(config, env_id, seed)
    layout["pair_index"] = episode_index % len(PAIRS)
    layout["instruction"] = INSTRUCTION
    return layout


def measured_poses(snapshot, base_world, kinds):
    """Convert each environment's actual rigid-body origins to its robot base."""
    base = np.asarray(base_world, dtype=float)
    if base.shape != (3,) or not np.isfinite(base).all():
        raise ValueError("base_world must be a finite world XYZ position")
    poses = {}
    for kind in kinds:
        obj = snapshot["objects"][kind]
        xyz = np.asarray(obj["position_world"], dtype=float)
        quat = np.asarray(obj["orientation_wxyz"], dtype=float)
        if (obj.get("active") is not True or xyz.shape != (3,) or quat.shape != (4,)
                or not np.isfinite(xyz).all() or not np.isfinite(quat).all()
                or np.linalg.norm(quat) < 1e-12):
            raise ValueError(f"Invalid settled observation for {kind}")
        poses[kind] = dict(position_base_m=(xyz-base).tolist(),
                           orientation_wxyz=(quat/np.linalg.norm(quat)).tolist())
    return poses


def plan_episode(normalized_config, reset_spec, snapshot, base_world, *,
                 trajectory_profile="compact", **planning_options):
    """Plan from the measured settled scene, with centered gamepad entry."""
    config = deepcopy(normalized_config)
    selected = tuple(reset_spec["active_names"])
    if len(selected) != 2 or len(set(selected)) != 2 or set(selected)-set(KINDS):
        raise ValueError("A cleanup demonstration needs exactly two different objects")
    for kind in selected:
        config["objects"][kind].update(deepcopy(reset_spec["objects"][kind]))
    poses = measured_poses(snapshot, base_world, selected)
    openings = {kind: .070 for kind in selected if kind == "speaker"} if trajectory_profile=="legacy" else {}
    options = dict(grasp_openings_m=openings, object_poses_base=poses,
                   trajectory_profile=trajectory_profile)
    options.update(planning_options)
    if options.get("object_poses_base") != poses:
        raise ValueError("Collection must use the measured settled object poses")
    plan = plan_cleanup(config, kinds=selected, **options)
    plan["collection"] = dict(instruction=INSTRUCTION, pair=list(selected),
        pose_source="measured_after_settling", pair_index=PAIRS.index(tuple(k for k in KINDS if k in selected)),
        reset_objects=deepcopy(reset_spec["objects"]))
    return plan


def assess_episode(history, plan):
    """Accept only complete, stable physical placement of both lifted objects."""
    if not history:
        return dict(passed=False, reason="no physical samples")
    final = history[-1]
    compact=plan.get("trajectory_profile","legacy")=="compact"
    validation=[]
    if compact:
        for row in reversed(history):
            if row["phase"]!="validation":break
            validation.append(row)
        validation.reverse()
    motion=history[:-len(validation)] if validation else history
    last_motion=motion[-1] if motion else None
    required_validation=max(1,int(np.ceil(float(plan.get("validation_hold_seconds",.6))*plan["fps"]))) if compact else 0
    validation_complete=not compact or (
        len(validation)>=required_validation
        and all(row["action_index"]==len(plan["actions"])-1
                and row.get("validation_index")==index for index,row in enumerate(validation)))
    physical = final["info"].get("physical_outcome", {})
    count = max(2, round(plan["fps"]*.5))
    window = (validation if validation else history)[-count:]
    outcomes = {}
    for kind in plan["kinds"]:
        positions = np.asarray([row["objects"][kind]["position_world"] for row in history])
        stable = np.asarray([row["objects"][kind]["position_world"] for row in window])
        quats = np.asarray([row["objects"][kind]["orientation_wxyz"] for row in window])
        rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]])
        state = physical.get("objects", {}).get(kind, {})
        measured = dict(max_lift_m=float(np.max(positions[:, 2])-positions[0, 2]),
            stable_position_excursion_m=float(np.max(np.linalg.norm(stable-stable[0], axis=1))),
            stable_rotation_excursion_rad=float(np.max((rotations[0].inv()*rotations).magnitude())),
            physical_outcome=deepcopy(state))
        measured["passed"] = bool(len(window) >= count and measured["max_lift_m"] > .025
            and measured["stable_position_excursion_m"] < .002
            and measured["stable_rotation_excursion_rad"] < np.deg2rad(4.)
            and all(state.get(key) is True for key in
                ("active", "grasped_and_lifted", "contained", "supported", "released", "low_velocity", "no_table_support")))
        outcomes[kind] = measured
    state = np.asarray(final["state"])
    full = bool(last_motion is not None and last_motion["action_index"] == len(plan["actions"])-1
                and last_motion["phase"] == (plan["phases"][-1] if compact else "settled"))
    home_error = float(np.max(np.abs(state[:6]-plan["expected_initial_state"][:6])))
    target=plan["actions"][-1][:6] if compact else plan["expected_initial_state"][:6]
    final_error=float(np.max(np.abs(state[:6]-target)))
    result = dict(full_plan_executed=full, terminal_validation_complete=bool(validation_complete),
        terminal_validation_frames=len(validation),required_validation_frames=required_validation,
        task_success=final["info"].get("success") is True,
        physical_success=physical.get("success") is True, objects=outcomes,
        final_gripper_width_m=float(state[6]), max_home_joint_error_rad=home_error,
        max_final_target_joint_error_rad=final_error)
    result["passed"] = bool(full and validation_complete and result["task_success"] and result["physical_success"]
        and all(row["passed"] for row in outcomes.values()) and state[6] > .040 and final_error < .030
        and not final.get("truncated", False))
    return result
