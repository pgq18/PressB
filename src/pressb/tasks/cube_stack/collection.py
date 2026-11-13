"""Task-specific planning and physical acceptance for stacking demonstrations.

The shared collector owns stepping, recording, perturbations and LeRobot export.
This adapter only chooses task parameters and targets from settled observations.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .planning import plan_stack
from .task import INSTRUCTION


def _merge(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = deepcopy(value)


def collection_config(base_config=None):
    """Return raw backend options; callers may choose GPU and parallelism."""
    config = dict(
        scene_config="configs/calibration/playground_20261113_appearance.json",
        num_envs=8, gpu=1, horizon=1, headless=True, max_episode_seconds=90.,
        terminate_on_success=False, settling_ticks=120, spacing_m=6.,
        render_resolution=[640,480], image_output_resolution=[640,480],
        observation_render_frames=1, reset_render_frames=2,
        smoothing={"mode":"none"}, cpu_threads=8, png_workers=8,
        initial_pose=dict(mode="workspace_center",height_above_table_m=.16,
                          radial_tilt_deg=15.,tool_yaw_deg=180.,gripper_width_m=.1),
        randomization=dict(enabled=True, x_range_m=[.28,.40], y_range_m=[-.14,.14],
                           yaw_range_deg=[-45.,45.], min_center_distance_m=.08, max_attempts=2048),
        log_dir="outputs/cube_stack_collection/simulation")
    if base_config is not None:
        source = base_config.get("backend",{}).get("config",base_config)
        if not isinstance(source,dict):
            raise ValueError("Collection base configuration must be a mapping")
        _merge(config,source)
    # Run through release/withdrawal and the separate terminal stability audit.
    config["terminate_on_success"]=False
    config["max_episode_seconds"]=max(90.,float(config["max_episode_seconds"]))
    return config


def plan_episode(normalized_config, reset_spec, snapshot, base_world, *,
                 lift_height_above_table=.080, max_joint_speed_deg_s=20.,
                 cartesian_speed_m_s=.040, trajectory_profile="compact"):
    """Plan from measured XY/yaw after the cubes settle on the tabletop.

    Cubes are invariant to 90-degree yaw rotations. Alternate parallel-jaw
    grasps use this symmetry without moving or relabelling the physical cube.
    A noticeably tilted or unsupported cube is rejected instead of pretending
    that a vertical grasp has been observed. Full-path IK failure remains a
    rejected attempt for the collector to log and resample.
    """
    base=np.asarray(base_world,float)
    if base.shape!=(3,) or not np.isfinite(base).all():
        raise ValueError("base_world must be a finite XYZ position")
    scene=deepcopy(normalized_config["scene"])
    objects={obj["name"]:obj for obj in scene["objects"]}
    measured={}
    mount=float(scene.get("robot_base_height_above_table_m",0.))
    for color in ("red","blue"):
        state=snapshot["objects"][color]
        position=np.asarray(state["position_world"],float)-base
        quaternion=np.asarray(state["orientation_wxyz"],float)
        if (position.shape!=(3,) or quaternion.shape!=(4,) or not np.isfinite(position).all()
                or not np.isfinite(quaternion).all() or not np.isclose(np.linalg.norm(quaternion),1.,atol=1e-5)):
            raise ValueError(f"Invalid settled pose for {color}")
        rotation=Rotation.from_quat(quaternion[[1,2,3,0]]).as_matrix()
        edge=float(normalized_config["cubes"][color]["edge_m"])
        bottom=position[2]-float(np.sum(np.abs(rotation[2])))*edge/2
        if rotation[2,2]<math.cos(math.radians(3.)) or abs(bottom+mount)>.003:
            raise ValueError(f"Settled {color} cube is not upright and tabletop-supported")
        yaw=math.degrees(math.atan2(rotation[1,0],rotation[0,0]))
        objects[color].update(center_xy=position[:2].tolist(),yaw_deg=yaw)
        measured[color]=dict(position_base_m=position.tolist(),orientation_wxyz=quaternion.tolist(),yaw_deg=yaw,
                             bottom_height_above_table_m=bottom+mount)
    source_yaw=objects["red"]["yaw_deg"]
    # Prefer the equivalent orientation nearest the established vertical grip.
    canonical=(source_yaw+45.)%90.-45.
    errors=[]
    for yaw in (canonical,canonical+90.,canonical-90.,canonical+180.):
        candidate=deepcopy(scene)
        source=next(obj for obj in candidate["objects"] if obj["name"]=="red")
        source["yaw_deg"]=yaw
        try:
            plan=plan_stack(candidate,lift_height_above_table=lift_height_above_table,
                max_joint_speed_deg_s=max_joint_speed_deg_s,cartesian_speed_m_s=cartesian_speed_m_s,
                trajectory_profile=trajectory_profile)
        except ValueError as exc:
            errors.append(str(exc)); continue
        plan.update(task_id="cube_stack",instruction=INSTRUCTION,
            planning_pose_source="measured_settled_xy_yaw",measured_object_poses_base=measured,
            reset_spec=deepcopy(reset_spec),grasp_equivalent_cube_yaw_deg=yaw,
            source_observed_yaw_deg=source_yaw,grasp_symmetry_quarter_turns=int(round((yaw-source_yaw)/90.)),
            rejected_grasp_candidates=errors)
        return plan
    raise ValueError("No complete bounded stack plan for settled layout: "+"; ".join(errors))


def assess_episode(history, plan):
    """Require the task's contact-supported success and a complete stable demo."""
    if not history:
        return dict(passed=False,reason="no physical samples")
    final=history[-1]
    compact=plan.get("trajectory_profile","legacy")=="compact"
    validation=[]
    if compact:
        for row in reversed(history):
            if row["phase"]!="validation":break
            validation.append(row)
        validation.reverse()
    motion=history[:-len(validation)] if validation else history
    last_motion=motion[-1] if motion else None
    required_validation=max(1,math.ceil(float(plan.get("validation_hold_seconds",.6))*plan["fps"])) if compact else 0
    validation_complete=not compact or (
        len(validation)>=required_validation
        and all(row["action_index"]==len(plan["actions"])-1
                and row.get("validation_index")==index for index,row in enumerate(validation)))
    full=bool(last_motion is not None and last_motion["action_index"]==len(plan["actions"])-1
              and last_motion["phase"]==(plan["phases"][-1] if compact else "settled"))
    physical=final["info"].get("physical_outcome",{})
    source_height,target_height=plan["source"]["size"][2],plan["target"]["size"][2]
    red=np.asarray([row["objects"]["red"]["position_world"] for row in history])
    blue=np.asarray(final["objects"]["blue"]["position_world"])
    tail=(validation if validation else history)[-max(2,round(plan["fps"]*.5)):]
    stable=np.asarray([row["objects"]["red"]["position_world"] for row in tail])
    rotations=Rotation.from_quat(np.asarray([row["objects"]["red"]["orientation_wxyz"] for row in tail])[:,[1,2,3,0]])
    state=np.asarray(final["state"])
    target=plan["actions"][-1][:6] if compact else plan["expected_initial_state"][:6]
    final_error=float(np.max(np.abs(state[:6]-target)))
    result=dict(full_plan_executed=full, terminal_validation_complete=bool(validation_complete),
        terminal_validation_frames=len(validation),required_validation_frames=required_validation,
        task_success=final["info"].get("success") is True,physical_success=physical.get("success") is True,
        physical_outcome=deepcopy(physical),max_red_lift_m=float(red[:,2].max()-red[0,2]),
        red_blue_xy_error_m=float(np.linalg.norm(red[-1,:2]-blue[:2])),
        center_height_separation_error_m=float(abs(red[-1,2]-blue[2]-(source_height+target_height)/2)),
        red_position_excursion_m=float(np.max(np.linalg.norm(stable-stable[0],axis=1))),
        red_rotation_excursion_rad=float(np.max((rotations[0].inv()*rotations).magnitude())),
        final_gripper_width_m=float(state[6]),
        max_final_target_joint_error_rad=final_error,
        max_home_joint_error_rad=float(np.max(np.abs(state[:6]-plan["expected_initial_state"][:6]))))
    result["passed"]=bool(result["full_plan_executed"] and validation_complete
        and result["task_success"] and result["physical_success"]
        and not final.get("truncated",False)
        and physical.get("grasped_and_lifted") is True and not physical.get("dropped",True)
        and result["max_red_lift_m"]>target_height+.010
        and result["red_blue_xy_error_m"]<min(plan["source"]["size"][:2])*.25
        and result["center_height_separation_error_m"]<.004
        and result["red_position_excursion_m"]<.001 and result["red_rotation_excursion_rad"]<math.radians(3)
        and state[6]>.050 and final_error<.030)
    return result
