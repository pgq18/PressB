"""CPU geometry and contact-supported cleanup outcome (SI units, wxyz poses)."""
from __future__ import annotations

import math
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import KINDS

INSTRUCTION = "Put all objects on the table into the black box."
DEFAULT_OUTCOME = dict(stable_seconds=.5, min_lift_height_m=.012, containment_tolerance_m=.001,
    support_gap_tolerance_m=.004, max_linear_speed_m_s=.025, max_angular_speed_rad_s=.25,
    gripper_release_width_m=.05, min_grasp_force_n=.02, min_support_force_n=.02,
    drop_below_table_m=.05, max_support_normal_tilt_deg=20., contact_separation_tolerance_m=.003)


def number(value, name, *, zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int, np.floating, np.integer)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def outcome_config(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value)-set(DEFAULT_OUTCOME):
        raise ValueError("Unknown cleanup outcome settings")
    cfg = {key: number(v, key) for key, v in {**DEFAULT_OUTCOME, **value}.items()}
    if cfg["max_support_normal_tilt_deg"] >= 90:
        raise ValueError("Support normal tolerance must be below 90 degrees")
    return cfg


def active_mask_array(value):
    mask = np.asarray(value)
    if mask.shape != (3,) or mask.dtype.kind != "b" or int(mask.sum()) != 2:
        raise ValueError("Cleanup requires exactly two active boolean object slots")
    return mask.copy()


class TabletopCleanupOutcome:
    """Every active object needs grasp/lift history and a support path to bin floor.

    Full surface vertices, rather than object centers or a bounding sphere, are
    tested in the bin frame. Stacked objects may exceed the rim, but their top
    cannot exceed the floor plus the two measured vertical spans (or rim).
    """
    def __init__(self, asset_specs, tabletop_z, bin_spec, active_mask, **settings):
        if not isinstance(asset_specs, (tuple, list)) or len(asset_specs) != 3:
            raise ValueError("asset_specs must follow the three fixed object slots")
        self.vertices = []
        for spec in asset_specs:
            vertices = np.asarray(spec["support_vertices"], dtype=float)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
                raise ValueError("Each asset needs finite full-surface support vertices")
            self.vertices.append(vertices.copy())
        self.active = active_mask_array(active_mask)
        if isinstance(tabletop_z, bool) or not math.isfinite(float(tabletop_z)):
            raise ValueError("tabletop_z must be finite")
        self.tabletop_z = float(tabletop_z)
        self.bin = deepcopy(bin_spec)
        self.bin_center = array(bin_spec["center_world"], (3,), "bin center")
        self.bin_lower = array(bin_spec["inner_min_local"], (3,), "bin inner minimum")
        self.bin_upper = array(bin_spec["inner_max_local"], (3,), "bin inner maximum")
        if np.any(self.bin_upper <= self.bin_lower):
            raise ValueError("Bin cavity must have positive dimensions")
        yaw = bin_spec.get("yaw_deg", 0.)
        if isinstance(yaw, bool) or not math.isfinite(float(yaw)):
            raise ValueError("Bin yaw must be finite")
        self.bin_rotation = Rotation.from_euler("z", float(yaw), degrees=True).as_matrix()
        self.floor = float(self.bin_center[2]+self.bin_lower[2])
        self.rim = float(self.bin_center[2]+self.bin_upper[2])
        self.config = outcome_config(settings)
        self.grasped_and_lifted = np.zeros(3, dtype=bool)
        self.max_lift_m = np.zeros(3)
        self.stable_since = self.previous_seconds = self.first_success_time_s = None
        self.ever_success = False

    def update(self, *, seconds, positions, quaternions_wxyz, velocities, gripper_width_m, contacts):
        seconds = number(seconds, "seconds", zero=True)
        if self.previous_seconds is not None and seconds <= self.previous_seconds:
            raise ValueError("Episode seconds must increase strictly")
        positions = array(positions, (3, 3), "positions")
        quats = array(quaternions_wxyz, (3, 4), "quaternions_wxyz")
        velocities = array(velocities, (3, 6), "velocities")
        if np.any(np.abs(np.linalg.norm(quats, axis=1)-1) > .01):
            raise ValueError("Object quaternions must be normalized")
        width = number(gripper_width_m, "gripper_width_m", zero=True)
        keys = {"finger_forces_n": (3, 2), "table_forces_n": (3,), "bin_floor_forces_n": (3,),
                "object_support_forces_n": (3, 3)}
        if not isinstance(contacts, dict) or set(contacts)-set(keys):
            raise ValueError("Unknown cleanup contact fields")
        forces = {key: array(contacts.get(key, np.zeros(shape)), shape, key) for key, shape in keys.items()}
        if any(np.any(value < 0) for value in forces.values()):
            raise ValueError("Contact force magnitudes must be nonnegative")
        if not np.allclose(forces["object_support_forces_n"], forces["object_support_forces_n"].T, atol=1e-12, rtol=0):
            raise ValueError("Object support contact forces must be symmetric")
        if np.any(np.diag(forces["object_support_forces_n"]) != 0):
            raise ValueError("An object cannot support itself")
        self.previous_seconds = seconds
        rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
        world = [v @ rotation.T + position for v, rotation, position in zip(self.vertices, rotations, positions)]
        bottoms = np.array([v[:, 2].min() for v in world])
        tops = np.array([v[:, 2].max() for v in world])
        cfg, ids = self.config, np.flatnonzero(self.active)
        tolerance = cfg["containment_tolerance_m"]
        max_top = max(self.rim, self.floor+float(np.sum((tops-bottoms)[ids])))
        contained, direct, supported = np.zeros(3, bool), np.zeros(3, bool), np.zeros(3, bool)
        pair = forces["object_support_forces_n"].copy()
        pair[~self.active, :], pair[:, ~self.active] = 0., 0.
        for i in ids:
            local = (world[i]-self.bin_center) @ self.bin_rotation
            contained[i] = bool(np.all(local[:, :2] >= self.bin_lower[:2]-tolerance) and
                np.all(local[:, :2] <= self.bin_upper[:2]+tolerance) and
                bottoms[i] >= self.floor-tolerance and tops[i] <= max_top+tolerance)
            direct[i] = bool(contained[i] and abs(float(bottoms[i]-self.floor)) <= cfg["support_gap_tolerance_m"]
                and forces["bin_floor_forces_n"][i] >= cfg["min_support_force_n"])
            self.max_lift_m[i] = max(self.max_lift_m[i], float(bottoms[i]-self.tabletop_z))
            unsupported = (forces["table_forces_n"][i] < cfg["min_support_force_n"] and
                forces["bin_floor_forces_n"][i] < cfg["min_support_force_n"] and
                not np.any(pair[i] >= cfg["min_support_force_n"]))
            if (np.all(forces["finger_forces_n"][i] >= cfg["min_grasp_force_n"]) and unsupported and
                    bottoms[i]-self.tabletop_z >= cfg["min_lift_height_m"]):
                self.grasped_and_lifted[i] = True
        supported[:] = direct
        # Three fixed slots, exactly two active: at most two graph expansions.
        for _ in ids:
            for upper in ids:
                for lower in ids:
                    if (upper != lower and supported[lower] and contained[upper] and
                        positions[upper, 2] > positions[lower, 2] and
                        abs(float(bottoms[upper]-tops[lower])) <= cfg["support_gap_tolerance_m"] and
                        pair[upper, lower] >= cfg["min_support_force_n"]):
                        supported[upper] = True
        objects = {}
        valid = True
        dropped = False
        for i, kind in enumerate(KINDS):
            if not self.active[i]:
                objects[kind] = {"active": False}
                continue
            released = bool(width >= cfg["gripper_release_width_m"] and
                np.all(forces["finger_forces_n"][i] < cfg["min_grasp_force_n"]))
            slow = bool(np.linalg.norm(velocities[i, :3]) <= cfg["max_linear_speed_m_s"] and
                        np.linalg.norm(velocities[i, 3:]) <= cfg["max_angular_speed_rad_s"])
            no_table = bool(forces["table_forces_n"][i] < cfg["min_support_force_n"])
            row = dict(active=True, grasped_and_lifted=bool(self.grasped_and_lifted[i]), contained=bool(contained[i]),
                supported=bool(supported[i]), direct_floor_support=bool(direct[i]), released=released,
                low_velocity=slow, no_table_support=no_table, bottom_world_m=float(bottoms[i]),
                top_world_m=float(tops[i]), max_lift_m=float(self.max_lift_m[i]))
            objects[kind] = row
            valid &= all(row[key] for key in ("grasped_and_lifted", "contained", "supported", "released", "low_velocity", "no_table_support"))
            dropped |= bool(tops[i] < self.tabletop_z-cfg["drop_below_table_m"])
        if not valid or dropped:
            self.stable_since = None
        elif self.stable_since is None:
            self.stable_since = seconds
        duration = 0. if self.stable_since is None else seconds-self.stable_since
        success = bool(valid and not dropped and duration+1e-12 >= cfg["stable_seconds"])
        event = bool(success and not self.ever_success)
        if event:
            self.ever_success, self.first_success_time_s = True, seconds
        return dict(success=success, success_event=event, ever_success=bool(self.ever_success), dropped=bool(dropped),
            objects=objects, active_mask=self.active.tolist(), stable_seconds=float(duration),
            max_allowed_top_world_m=float(max_top), first_success_time_s=self.first_success_time_s)
