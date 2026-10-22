"""Pure NumPy trajectory loading and physical pick-and-place outcome checks."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.spatial.transform import Rotation


def load_trajectory(path, source="observed_arm_action_gripper", maximum_width_m=.1):
    """Load audited NPZ; raw source angles are degrees and width is centimetres.

    Width saturation is explicit and returned as metadata. Recorded observations
    set the initial state, while replay selects either commands or observations.
    No joint offsets are invented and no arm angle is silently clipped.
    """
    if source not in ("action", "observation", "observed_arm_action_gripper"):
        raise ValueError("Unknown trajectory source")
    with np.load(path, allow_pickle=False) as data:
        state = np.asarray(data["state_raw"], dtype=float)
        action = np.asarray(data["action_raw"], dtype=float)
        times = np.asarray(data["timestamp_s"], dtype=float)
    if state.ndim != 2 or state.shape[1] != 7 or action.shape != state.shape:
        raise ValueError("Expected paired Nx7 joint/gripper records")
    if times.shape != (len(state),) or len(times) < 2 or not np.isfinite([state, action]).all():
        raise ValueError("Invalid trajectory shape or values")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing")
    if not 0 < maximum_width_m <= .1:
        raise ValueError("Invalid gripper travel")
    selected = action if source == "action" else state
    if source == "observed_arm_action_gripper":
        selected = np.column_stack((state[:, :6], action[:, 6]))
    widths = selected[:, 6] * .01
    commands = np.column_stack((np.deg2rad(selected[:, :6]), np.clip(widths, 0, maximum_width_m)))
    initial = np.r_[np.deg2rad(state[0, :6]), np.clip(state[0, 6] * .01, 0, maximum_width_m)]
    return dict(times=times - times[0], commands=commands, initial=initial,
                metadata=dict(source=source, source_angle_unit="degree", source_width_unit="cm",
                              arm_source="action" if source == "action" else "observation.state",
                              gripper_source="observation.state" if source == "observation" else "action",
                              unit_verified_from_capture_source=False,
                              width_scale_m=.01,
                              width_saturation_rule="Clip negative closure commands to zero and openings to configured travel; retain raw input",
                              width_unit_status="inferred from ranges and measured grasp width; source metadata omits units",
                              maximum_width_m=maximum_width_m,
                              width_saturated_samples=int(np.count_nonzero(widths != commands[:, 6])),
                              max_width_saturation_m=float(np.max(np.abs(widths - commands[:, 6]))),
                              raw_width_range_m=[float(widths.min()), float(widths.max())]))


def object_geometry(value):
    """Normalize a legacy cube edge or a supported rigid primitive's geometry."""
    if isinstance(value, Mapping):
        result = dict(value)
        shape = result.get("shape")
        if shape == "cube":
            if "edge" in result and result["edge"] is not None:
                edge = float(result["edge"])
                if not np.isfinite(edge) or edge <= 0:
                    raise ValueError("Cube edge must be finite and positive")
                size = np.full(3, edge)
            else:
                size = np.asarray(result.get("size_xyz", result.get("size")), dtype=float)
                if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
                    raise ValueError("Cube dimensions must be three finite positive equal values")
                edge = float(size[0])
            for key in ("size", "size_xyz"):
                if key in result:
                    supplied = np.asarray(result[key], dtype=float)
                    if supplied.shape != (3,) or not np.allclose(supplied, edge, rtol=1e-9, atol=1e-12):
                        raise ValueError("Cube dimensions must equal its single edge on all three axes")
            result["edge"] = edge
            size = np.full(3, edge)
        elif shape == "cuboid":
            size = np.asarray(result.get("size_xyz", result.get("size")), dtype=float)
        elif shape == "cylinder":
            radius, height = float(result["radius"]), float(result["height"])
            size = np.array([2*radius, 2*radius, height])
        else:
            raise ValueError(f"Unsupported object shape: {shape}")
    else:
        edge = float(value)
        result, size = dict(shape="cuboid"), np.full(3, edge)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("Object dimensions must be three finite positive values")
    result["size_xyz"] = size.tolist()
    result["grasp_width_m"] = float(result.get("grasp_width_m", min(size[:2])))
    if not np.isfinite(result["grasp_width_m"]) or result["grasp_width_m"] <= 0:
        raise ValueError("Object grasp width must be finite and positive")
    return result


def _half_extent(rotation, geometry, directions):
    """Exact support of a centred cuboid/cylinder along each requested axis."""
    local = np.asarray(directions) @ rotation
    if geometry["shape"] == "cylinder":
        return (float(geometry["radius"]) * np.linalg.norm(local[..., :2], axis=-1)
                + float(geometry["height"])/2 * np.abs(local[..., 2]))
    return np.abs(local) @ (np.asarray(geometry["size_xyz"])/2)


def object_inside_box(position, quaternion_wxyz, geometry, box):
    """Require the entire rotated primitive inside the open box's inner volume.

    Analytic support distances test all six cavity boundaries. A cylinder uses
    its curved surface, rather than the corners of its enclosing cuboid.
    """
    position = np.asarray(position, dtype=float)
    quat = np.asarray(quaternion_wxyz, dtype=float)
    if position.shape != (3,) or quat.shape != (4,) or not np.isfinite(np.r_[position, quat]).all():
        return False
    if abs(np.linalg.norm(quat)-1) > .01:
        return False
    geometry = object_geometry(geometry)
    rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    center = np.asarray(box["center_world"])
    inverse = Rotation.from_euler("z", -float(box.get("yaw_deg", 0)), degrees=True).as_matrix()
    local = inverse @ (position-center)
    extent = _half_extent(rotation, geometry, inverse)
    half = np.asarray(box["inner_size_xy"]) / 2
    return bool(np.all(np.abs(local[:2]) + extent[:2] <= half + .001) and
                position[2]-extent[2] >= float(box["floor_z_world"]) - .002 and
                position[2]+extent[2] <= float(box["rim_z_world"]) + .001)


def cube_inside_box(position, quaternion_wxyz, edge, box):
    """Backward-compatible cube predicate, with the original geometric tolerances."""
    if not np.isfinite(edge) or edge <= 0:
        return False
    return object_inside_box(position, quaternion_wxyz, edge, box)


class PickPlaceOutcome:
    """Success requires lift, release, and the entire object resting in the box."""
    def __init__(self, edge, tabletop_z, box, stable_seconds=.5, lift_height=.04):
        self.geometry = object_geometry(edge)
        self.edge = self.geometry["grasp_width_m"]
        self.tabletop_z, self.box = tabletop_z, box
        self.stable_seconds, self.lift_height = stable_seconds, lift_height
        self.max_lift_m = 0.
        self.max_bottom_clearance_m = 0.
        self.stable_since = None
        self.success = False
        self.first_success_time_s = None

    def update(self, seconds, position, quaternion, velocity, gripper_width, finger_contact=False):
        self.max_lift_m = max(self.max_lift_m, float(position[2])-self.tabletop_z-self.geometry["size_xyz"][2]/2)
        inside = object_inside_box(position, quaternion, self.geometry, self.box)
        velocity = np.asarray(velocity)
        released = gripper_width >= self.edge + .008 and not finger_contact
        rotation = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
        bottom = float(position[2]) - float(_half_extent(rotation, self.geometry, [0., 0., 1.]))
        # Tipping a long object upright can raise its centre while it remains
        # supported by the table. Actual lift requires its lowest point clear.
        self.max_bottom_clearance_m = max(self.max_bottom_clearance_m, bottom-self.tabletop_z)
        lifted = self.max_bottom_clearance_m >= self.lift_height
        resting_on_box_floor = abs(bottom-float(self.box["floor_z_world"])) <= .003
        stable = np.linalg.norm(velocity[:3]) < .025 and np.linalg.norm(velocity[3:]) < .25
        valid = inside and released and stable and resting_on_box_floor and lifted
        if not valid:
            self.stable_since = None
        elif self.stable_since is None:
            self.stable_since = float(seconds)
        if valid and seconds-self.stable_since >= self.stable_seconds and not self.success:
            self.success = True
            self.first_success_time_s = float(seconds)
        return dict(inside_box=inside, gripper_released=released, object_stable=bool(stable),
                    resting_on_box_floor=resting_on_box_floor, finger_contact=bool(finger_contact),
                    lifted=lifted, max_lift_m=self.max_lift_m,
                    max_bottom_clearance_m=self.max_bottom_clearance_m,
                    success=self.success, first_success_time_s=self.first_success_time_s)
