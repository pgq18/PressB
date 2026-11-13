"""Resolve a PiPER starting pose over a configured tabletop work area on CPU."""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import PiperKinematics


def apply_workspace_home(scene, region, options=None, *, explicit_joint_override=False):
    """Update only the three initial-state aliases; preserve all camera settings.

    An explicit scene joint pose remains authoritative unless the caller also
    explicitly selects ``workspace_center``. ``scene`` reproduces historical
    starts. The centre uses configured spawn bounds, never sampled object poses.
    Unreachable requests fail instead of silently moving the requested centre.
    """
    options = {} if options is None else deepcopy(options)
    allowed = {"mode", "height_above_table_m", "radial_tilt_deg", "tool_yaw_deg", "gripper_width_m"}
    if not isinstance(options, dict) or set(options)-allowed:
        raise ValueError(f"initial_pose supports only {sorted(allowed)}")
    mode = options.get("mode", "scene" if explicit_joint_override else "workspace_center")
    if mode not in ("scene", "workspace_center"):
        raise ValueError("initial_pose.mode must be scene or workspace_center")
    if mode == "scene":
        if set(options)-{"mode"}:
            raise ValueError("initial_pose scene mode does not accept Cartesian options")
        return dict(mode=mode, state_joint_rad_m=deepcopy(scene["initial_state_joint_rad_m"]))
    height = options.get("height_above_table_m", .16)
    tilt = options.get("radial_tilt_deg", 15.)
    yaw = options.get("tool_yaw_deg", 180.)
    width = options.get("gripper_width_m", scene["gripper_max_width_m"])
    for name, value in (("height_above_table_m",height), ("radial_tilt_deg",tilt),
                        ("tool_yaw_deg",yaw), ("gripper_width_m",width)):
        if isinstance(value, bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError(f"initial_pose.{name} must be finite")
    if height <= 0 or not 0 <= tilt < 90 or not 0 <= width <= scene["gripper_max_width_m"]:
        raise ValueError("initial_pose height, tilt or gripper width is outside bounds")
    bounds = np.asarray([region["x_range_m"],region["y_range_m"]],dtype=float)
    if bounds.shape != (2,2) or not np.isfinite(bounds).all() or np.any(bounds[:,1] <= bounds[:,0]):
        raise ValueError("Workspace centre requires finite increasing X/Y bounds")
    xy = bounds.mean(axis=1)
    radius = float(np.linalg.norm(xy))
    if radius < 1e-6:
        raise ValueError("Workspace centre must be away from the robot base")
    tangent = np.array([-xy[1],xy[0],0.])/radius
    down = Rotation.from_euler("z",yaw,degrees=True).as_matrix()@np.diag([1.,-1.,-1.])
    rotation = Rotation.from_rotvec(-tangent*math.radians(tilt)).as_matrix()@down
    target = np.r_[xy,float(height)-float(scene.get("robot_base_height_above_table_m",0.))]
    kin = PiperKinematics(scene["robot_urdf"],tip_offset=.1358)
    q = kin.solve(target,np.asarray(scene["initial_state_joint_rad_m"][:6],float),rotation)
    state = np.r_[q,float(width)]
    if np.any(state < np.r_[kin.lower,0.]) or np.any(state > np.r_[kin.upper,scene["gripper_max_width_m"]]):
        raise ValueError("Centred initial pose exceeds robot joint/gripper limits")
    scene["initial_state_joint_rad_m"] = state.tolist()
    scene["initial_joint_positions_rad"] = q.tolist()
    scene["initial_gripper_width_m"] = float(width)
    actual = kin.fk(q)
    return dict(mode=mode,height_above_table_m=float(height),radial_tilt_deg=float(tilt),
                tool_yaw_deg=float(yaw),gripper_width_m=float(width),
                center_xy_m=xy.tolist(),tcp_target_base_m=target.tolist(),
                tcp_actual_base_m=actual[:3,3].tolist(),tool_rotation_base=rotation.tolist(),
                state_joint_rad_m=state.tolist(),
                min_joint_limit_margin_deg=float(np.rad2deg(np.minimum(q-kin.lower,kin.upper-q).min())))
