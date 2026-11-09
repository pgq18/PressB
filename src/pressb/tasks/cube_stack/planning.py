"""CPU-only joint plan for a physical red-on-blue cube stack.

Uses the playground grasp TCP definition and explicit table-relative heights.
The plan contains only PiPER joint/gripper targets: no object poses, attachment
constraints or simulator stepping operations are executed here.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pressb.robots.piper.kinematics import PiperKinematics

ROOT = Path(__file__).resolve().parents[4]


def plan_stack(scene, *, source_name="red", target_name="blue",
               grasp_height_above_table=.012, lift_height_above_table=.090,
               release_gap=.001, open_width=.060, close_width=.026,
               max_joint_speed_deg_s=20., gripper_speed_m_s=.030,
               cartesian_speed_m_s=.040, align_target_yaw=True):
    """Make an inspectable, bounded approach/grasp/lift/place/release/return plan."""
    objects = {obj["name"]: obj for obj in scene["objects"]}
    source, target = objects[source_name], objects[target_name]
    initial = np.asarray(scene["initial_state_joint_rad_m"], dtype=float)
    size, target_size = np.asarray(source["size"],dtype=float), np.asarray(target["size"],dtype=float)
    if initial.shape!=(7,) or size.shape!=(3,) or target_size.shape!=(3,) or not np.isfinite(
            np.r_[initial,size,target_size]).all() or np.any(size<=0) or np.any(target_size<=0):
        raise ValueError("Initial joints and two positive object sizes are required")
    values = [grasp_height_above_table,lift_height_above_table,release_gap,open_width,close_width,
              max_joint_speed_deg_s,gripper_speed_m_s,cartesian_speed_m_s]
    if not np.isfinite(values).all() or min(values)<=0:
        raise ValueError("Stack planning lengths and speeds must be finite and positive")
    if not close_width < min(size[:2]) < open_width <= scene["gripper_max_width_m"]:
        raise ValueError("Gripper opening must bracket the source cube width")
    if not 0 < grasp_height_above_table < size[2] or lift_height_above_table <= target_size[2]+size[2]+.02:
        raise ValueError("Grasp/lift heights do not clear the objects")
    mount = float(scene.get("robot_base_height_above_table_m",0.))
    if not math.isfinite(mount) or mount<0:
        raise ValueError("Invalid robot mounting height")
    urdf = Path(scene["robot_urdf"])
    kin = PiperKinematics(urdf if urdf.is_absolute() else ROOT/urdf,tip_offset=.1358)
    fps = float(scene["fps"])
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError("Plan FPS must be positive and finite")
    xy = np.asarray(source["center_xy"],dtype=float)
    destination = np.asarray(target["center_xy"],dtype=float)
    if xy.shape!=(2,) or destination.shape!=(2,) or not np.isfinite(np.r_[xy,destination]).all():
        raise ValueError("Object centres must be finite XY positions")
    grasp_yaw = float(source.get("yaw_deg",0.))+180.
    yaw_delta = (float(target.get("yaw_deg",0.))-float(source.get("yaw_deg",0.))+45.)%90.-45.
    placement_yaw = grasp_yaw+(yaw_delta if align_target_yaw else 0.)
    yaw = grasp_yaw
    def orientation(angle):
        return Rotation.from_euler("z",angle,degrees=True).as_matrix()@np.diag([1.,-1.,-1.])
    grasp_z = grasp_height_above_table-mount
    high_z = lift_height_above_table-mount
    placement_z = grasp_z+target_size[2]+release_gap
    speed = np.r_[np.minimum(kin.velocity_limits,np.deg2rad(max_joint_speed_deg_s)),gripper_speed_m_s]
    waypoints = []
    q = initial[:6].copy()
    current_tcp = kin.fk(q)[:3,3]

    def add(phase, state, hold=0., minimum_duration=0.):
        nonlocal q,current_tcp
        q = np.asarray(state[:6]).copy()
        current_tcp = kin.fk(q)[:3,3]
        waypoints.append(dict(phase=phase,action=np.asarray(state).tolist(),hold_seconds=hold,
            minimum_duration_s=minimum_duration,tcp_base_m=current_tcp.tolist()))

    def cartesian(phase, position, width, hold=0., end_yaw=None):
        nonlocal yaw
        start = current_tcp.copy()
        position = np.asarray(position,dtype=float)
        start_yaw, end_yaw = yaw, yaw if end_yaw is None else end_yaw
        count = max(1,int(math.ceil(np.linalg.norm(position-start)/.004)),int(math.ceil(abs(end_yaw-start_yaw)/4.)))
        for fraction,point in zip(np.arange(1,count+1)/count,np.linspace(start,position,count+1)[1:]):
            distance = float(np.linalg.norm(point-current_tcp))
            yaw = start_yaw+(end_yaw-start_yaw)*fraction
            solved = kin.solve(point,q,orientation(yaw))
            add(phase,np.r_[solved,width],minimum_duration=distance/cartesian_speed_m_s)
        waypoints[-1]["hold_seconds"] = hold

    add("open",np.r_[q,open_width],.5)
    q = kin.solve(np.r_[xy,high_z],q,orientation(grasp_yaw))
    add("pregrasp",np.r_[q,open_width],.5)
    cartesian("descend",np.r_[xy,grasp_z],open_width,.5)
    add("close",np.r_[q,close_width],1.)
    cartesian("lift",np.r_[xy,high_z],close_width,.6)
    cartesian("transfer",np.r_[destination,high_z],close_width,.4,end_yaw=placement_yaw)
    cartesian("place",np.r_[destination,placement_z],close_width,.6)
    add("release",np.r_[q,open_width],1.)
    cartesian("retreat",np.r_[destination,high_z],open_width,.5)
    add("return_home",np.r_[initial[:6],open_width],1.)
    add("settled",np.r_[initial[:6],open_width],2.)

    actions,phases = [],[]
    previous = initial.copy()
    for waypoint in waypoints:
        command = np.asarray(waypoint["action"])
        seconds = max(float(np.max(np.abs(command-previous)/speed)),waypoint["minimum_duration_s"])
        frames = max(1,int(math.ceil(seconds*fps)))
        waypoint["start_frame"] = len(actions)
        for fraction in np.arange(1,frames+1)/frames:
            actions.append(previous+(command-previous)*fraction)
            phases.append(waypoint["phase"])
        waypoint["arrival_frame"] = len(actions)-1
        for _ in range(round(waypoint["hold_seconds"]*fps)):
            actions.append(command.copy());phases.append(waypoint["phase"])
        waypoint["end_frame"] = len(actions)-1
        previous = command
    actions = np.asarray(actions)
    limits_low,limits_high = np.r_[kin.lower,0.],np.r_[kin.upper,scene["gripper_max_width_m"]]
    if np.any(actions<limits_low-1e-10) or np.any(actions>limits_high+1e-10):
        raise ValueError("Stack plan exceeds URDF joint or gripper limits")
    measured_speed = np.max(np.abs(np.diff(np.vstack([initial,actions]),axis=0)),axis=0)*fps
    if np.any(measured_speed>speed+1e-8):
        raise ValueError("Stack plan exceeds its declared speed limits")
    tcp = np.array([kin.fk(row[:6])[:3,3] for row in actions])
    if np.min(tcp[:,2])<grasp_z-.0001:
        raise ValueError("Interpolated TCP dips below the reviewed grasp height")
    return dict(schema_version=1,simulation_only=True,hardware_access=False,
        source_name=source_name,target_name=target_name,source=deepcopy(source),target=deepcopy(target),
        expected_initial_state=initial.tolist(),fps=fps,actions=actions.tolist(),phases=phases,waypoints=waypoints,
        duration_s=len(actions)/fps,units=["rad"]*6+["m"],tip_offset_m=.1358,
        robot_base_height_above_table_m=mount,grasp_height_above_table_m=grasp_height_above_table,
        grasp_height_base_m=grasp_z,lift_height_above_table_m=lift_height_above_table,
        release_gap_m=release_gap,placement_tcp_height_base_m=float(placement_z),
        align_target_yaw=bool(align_target_yaw),transfer_yaw_rotation_deg=placement_yaw-grasp_yaw,
        open_width_m=open_width,close_width_m=close_width,max_joint_speed_deg_s=max_joint_speed_deg_s,
        measured_max_speed_rad_m_s=measured_speed.tolist(),min_tcp_base_z_m=float(np.min(tcp[:,2])),
        object_motion="Only contact and gravity after reset; no attachment or teleportation")
