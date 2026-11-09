"""CPU-only contact pick/place plans for the two-object cleanup task.

This is an inspectable simulation demonstrator, not a learned policy or a
collision-complete motion planner. It emits only seven joint/gripper targets;
the task's dynamic objects are never attached or repositioned during playback.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from pressb.robots.piper.kinematics import PiperKinematics
from .geometry import asset_spec

ROOT = Path(__file__).resolve().parents[4]


def _rotation(yaw):
    return Rotation.from_euler("z", yaw, degrees=True).as_matrix()


def placement_slots(config, kinds, *, gap_m=.006, edge_margin_m=.003):
    """Pack the full visual envelopes, upright, along the bin's long axis."""
    if len(kinds) != 2 or len(set(kinds)) != 2:
        raise ValueError("Cleanup requires exactly two distinct object kinds")
    box = config["bin"]
    inner = np.asarray(box["size_xyz_m"], float)[:2]-2*float(box["wall_thickness_m"])
    specs = [asset_spec(k, config["objects"][k].get("scale", 1.)) for k in kinds]
    sizes = np.asarray([s["size"] for s in specs])
    span = sizes[:, 0].sum()+gap_m
    if span+2*edge_margin_m > inner[0]+1e-9 or np.max(sizes[:, 1])+2*edge_margin_m > inner[1]+1e-9:
        raise ValueError("The selected objects do not fit side by side inside this bin")
    # Put the larger footprint toward +X. Its centre then remains nearer the
    # bin centre, avoiding a tiny object's unnecessarily distant release pose.
    if sizes[0,0]>=sizes[1,0]:
        x = [span/2-sizes[0,0]/2, -span/2+sizes[1,0]/2]
    else:
        x = [-span/2+sizes[0,0]/2, span/2-sizes[1,0]/2]
    rz = _rotation(box.get("yaw_deg", 0.))
    return {k:dict(center_xy=(np.asarray(box["center_xy"])+rz[:2,:2]@np.array([xx,0.])).tolist(),
                   yaw_deg=float(box.get("yaw_deg", 0.))) for k,xx in zip(kinds,x)}


def plan_cleanup(config, *, kinds=("speaker", "lidar"), max_joint_speed_deg_s=20.,
                 gripper_speed_m_s=.030, cartesian_speed_m_s=.040,
                 lift_height_above_table_m=.120, contact_to_tip_m=.004,
                 release_gap_m=.007, grasp_openings_m=None):
    """Plan two sequential physical grasps and releases, then return home.

    The contact patch from each mesh is converted to the lowest fingertip TCP.
    The carry can tilt radially outward to keep J5 inside its existing URDF
    limit. Place/release is upright; impossible upright poses fail before Kit.
    Every IK solution uses the unchanged common solver tolerances and limits.
    """
    kinds = tuple(kinds)
    slots = placement_slots(config, kinds)
    if grasp_openings_m is None:
        grasp_openings_m = {}
    if not isinstance(grasp_openings_m, dict) or set(grasp_openings_m)-set(kinds):
        raise ValueError("Grasp opening overrides must map selected object kinds to widths in metres")
    values = [max_joint_speed_deg_s, gripper_speed_m_s, cartesian_speed_m_s,
              lift_height_above_table_m, contact_to_tip_m, release_gap_m]
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("Planning speeds, heights and clearances must be finite and positive")
    scene = config["scene"]
    initial = np.asarray(scene["initial_state_joint_rad_m"], float)
    fps = float(scene["fps"])
    mount = float(scene.get("robot_base_height_above_table_m", 0.))
    if initial.shape != (7,) or not np.isfinite(initial).all() or not math.isfinite(fps) or fps<=0:
        raise ValueError("A finite seven-dimensional initial state and positive FPS are required")
    urdf = Path(scene["robot_urdf"])
    kin = PiperKinematics(urdf if urdf.is_absolute() else ROOT/urdf, tip_offset=.1358)
    max_width = float(scene["gripper_max_width_m"])
    speed = np.r_[np.minimum(kin.velocity_limits, np.deg2rad(max_joint_speed_deg_s)), gripper_speed_m_s]
    q = initial[:6].copy()
    tcp = kin.fk(q)[:3,3]
    rot = kin.fk(q)[:3,:3]
    waypoints, grasps = [], {}
    floor = float(config["bin"]["wall_thickness_m"])
    box_height = float(config["bin"]["size_xyz_m"][2])
    if lift_height_above_table_m < box_height+.040:
        raise ValueError("Carry height must clear the box rim")

    def add(phase, state, hold=0., minimum_duration=0.):
        nonlocal q, tcp, rot
        q = np.asarray(state[:6]).copy()
        transform = kin.fk(q)
        tcp, rot = transform[:3,3], transform[:3,:3]
        waypoints.append(dict(phase=phase, action=np.asarray(state).tolist(), hold_seconds=hold,
                              minimum_duration_s=minimum_duration, tcp_base_m=tcp.tolist()))

    def cartesian(phase, destination, orientation, width, hold=0.):
        start, rotation = tcp.copy(), rot.copy()
        destination = np.asarray(destination, float)
        angle = (Rotation.from_matrix(rotation).inv()*Rotation.from_matrix(orientation)).magnitude()
        count = max(1, math.ceil(np.linalg.norm(destination-start)/.004), math.ceil(angle/math.radians(3.)))
        fractions = np.arange(1,count+1)/count
        rotations = Slerp([0.,1.], Rotation.from_matrix([rotation,orientation]))(fractions).as_matrix()
        for fraction, target_rot in zip(fractions, rotations):
            target = start+(destination-start)*fraction
            distance = float(np.linalg.norm(target-tcp))
            solved = kin.solve(target, q, target_rot)
            add(phase, np.r_[solved,width], minimum_duration=distance/cartesian_speed_m_s)
        waypoints[-1]["hold_seconds"] = hold

    for kind in kinds:
        spec = asset_spec(kind, config["objects"][kind].get("scale",1.))
        obj = deepcopy(config["objects"][kind])
        rz = _rotation(obj.get("yaw_deg",0.))
        object_center = np.r_[obj["center_xy"], spec["size"][2]/2-mount]
        candidate = spec["grasp_candidates"][0]
        open_width = float(grasp_openings_m.get(kind,candidate["opening_m"]))
        close_width = float(candidate["closed_width_m"])
        if not 0 < close_width < candidate["surface_width_m"] < open_width <= max_width:
            raise ValueError(f"{kind} grasp does not fit the configured gripper")
        axis = np.asarray(candidate["closing_axis_local"])
        # Tool Y is the closing axis. Sign is interchangeable for the two fingers.
        offset_angle = math.degrees(math.atan2(axis[1],axis[0]))+90.
        grasp_rot = _rotation(obj.get("yaw_deg",0.)+offset_angle)@np.diag([1.,-1.,-1.])
        place_rot = _rotation(slots[kind]["yaw_deg"]+offset_angle)@np.diag([1.,-1.,-1.])
        grasp_tcp = object_center+rz@candidate["center_local_m"]-np.array([0.,0.,contact_to_tip_m])
        if grasp_tcp[2]+mount < .003:
            raise ValueError(f"{kind} fingertip would approach within 3 mm of the table")
        # Desired upright root height preserves a small free-fall gap above the floor.
        place_root = np.r_[slots[kind]["center_xy"], floor+release_gap_m+spec["size"][2]/2-mount]
        root_in_tool = grasp_rot.T@(object_center-grasp_tcp)
        place_tcp = place_root-place_rot@root_in_tool
        high_pick = np.r_[grasp_tcp[:2],lift_height_above_table_m-mount]
        high_place = np.r_[place_tcp[:2],lift_height_above_table_m-mount]
        # Check low endpoints before the many intermediate solves.
        q_pick = kin.solve(grasp_tcp,q,grasp_rot)
        kin.solve(place_tcp,q_pick,place_rot)
        # The source may sit near the reach boundary. It only needs to clear
        # itself vertically; the subsequent tilted carry rises above the rim.
        minimum_pick_height = max(spec["size"][2]+.010,grasp_tcp[2]+mount+.025)
        for height in np.arange(lift_height_above_table_m, minimum_pick_height, -.015):
            high_pick[2] = height-mount
            try:
                q_pregrasp = kin.solve(high_pick,q_pick,grasp_rot)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"No vertical pregrasp clearing the source object for {kind}")
        radial = high_place[:2]/np.linalg.norm(high_place[:2])
        tangent = np.r_[-radial[1],radial[0],0.]
        carry_rot = None
        for tilt in (0.,5.,10.,15.,20.,25.,30.):
            trial = Rotation.from_rotvec(-tangent*math.radians(tilt)).as_matrix()@place_rot
            try:
                # Nine points rule out a branch that reaches only the endpoints.
                seed = q_pick.copy()
                for point in np.linspace(high_pick,high_place,9):
                    seed = kin.solve(point,seed,trial)
            except ValueError:
                continue
            carry_rot = trial
            break
        if carry_rot is None:
            raise ValueError(f"No bounded carry orientation found for {kind}")
        prefix = kind+":"
        add(prefix+"open", np.r_[q,open_width], .35)
        add(prefix+"pregrasp", np.r_[q_pregrasp,open_width], .35)
        cartesian(prefix+"descend",grasp_tcp,grasp_rot,open_width,.35)
        add(prefix+"close",np.r_[q,close_width],.8)
        cartesian(prefix+"lift",high_pick,grasp_rot,close_width,.4)
        cartesian(prefix+"orient_carry",high_pick,carry_rot,close_width,.2)
        cartesian(prefix+"transfer",high_place,carry_rot,close_width,.3)
        cartesian(prefix+"place",place_tcp,place_rot,close_width,.5)
        add(prefix+"release",np.r_[q,open_width],1.)
        cartesian(prefix+"retreat",high_place,carry_rot,open_width,.3)
        add(prefix+"return_home",np.r_[initial[:6],open_width],.5)
        grasps[kind] = dict(object=obj,asset_size_m=spec["size"],candidate=candidate,
                            grasp_tcp_base_m=grasp_tcp.tolist(),placement=slots[kind],
                            placement_tcp_base_m=place_tcp.tolist(),carry_tilt_deg=tilt,
                            open_width_m=open_width,close_width_m=close_width)
    add("settled",np.r_[initial[:6],max(g["open_width_m"] for g in grasps.values())],2.)
    actions, phases, previous = [], [], initial.copy()
    for waypoint in waypoints:
        command = np.asarray(waypoint["action"])
        duration = max(float(np.max(np.abs(command-previous)/speed)), waypoint["minimum_duration_s"])
        count = max(1,math.ceil(duration*fps))
        waypoint["start_frame"] = len(actions)
        for fraction in np.arange(1,count+1)/count:
            actions.append(previous+(command-previous)*fraction); phases.append(waypoint["phase"])
        waypoint["arrival_frame"] = len(actions)-1
        for _ in range(round(waypoint["hold_seconds"]*fps)):
            actions.append(command.copy()); phases.append(waypoint["phase"])
        waypoint["end_frame"] = len(actions)-1
        previous = command
    actions = np.asarray(actions)
    if np.any(actions < np.r_[kin.lower,0.]-1e-10) or np.any(actions > np.r_[kin.upper,max_width]+1e-10):
        raise ValueError("Cleanup plan exceeds existing URDF joint/gripper bounds")
    measured = np.max(np.abs(np.diff(np.vstack([initial,actions]),axis=0)),axis=0)*fps
    if np.any(measured > speed+1e-8):
        raise ValueError("Cleanup plan exceeds its speed limits")
    tcp_positions = np.asarray([kin.fk(row[:6])[:3,3] for row in actions])
    if tcp_positions[:,2].min()+mount < .002:
        raise ValueError("Interpolated fingertips approach within 2 mm of the table")
    return dict(schema_version=1,simulation_only=True,hardware_access=False,kinds=list(kinds),
                expected_initial_state=initial.tolist(),fps=fps,actions=actions.tolist(),phases=phases,
                waypoints=waypoints,grasps=grasps,bin=deepcopy(config["bin"]),duration_s=len(actions)/fps,
                units=["rad"]*6+["m"],tip_offset_m=.1358,robot_base_height_above_table_m=mount,
                max_joint_speed_deg_s=max_joint_speed_deg_s,measured_max_speed_rad_m_s=measured.tolist(),
                grasp_opening_overrides_m=deepcopy(grasp_openings_m),
                min_tcp_base_z_m=float(tcp_positions[:,2].min()),
                object_motion="Contact and gravity only after reset; no attachment or teleportation")
