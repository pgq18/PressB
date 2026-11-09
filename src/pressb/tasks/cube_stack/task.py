"""Measured cube stacking geometry and a contact-supported success predicate.

Positions/widths are metres, velocity is [m/s, rad/s], quaternions are wxyz,
forces are newtons and episode time is seconds. There is no simulator import.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation

INSTRUCTION = "Pick up the red cube and place it on top of the blue cube."
COLORS = ("red", "blue")
DEFAULT_OUTCOME = dict(stable_seconds=.5, min_lift_height_m=.012,
    min_xy_overlap_fraction=.75, max_tilt_deg=10., support_height_tolerance_m=.003,
    table_height_tolerance_m=.003, max_linear_speed_m_s=.025, max_angular_speed_rad_s=.25,
    gripper_release_margin_m=.008, min_grasp_force_n=.02, min_support_force_n=.02,
    drop_below_table_m=.04, max_support_normal_tilt_deg=20., contact_separation_tolerance_m=.003)


def _number(value, name, *, zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def outcome_config(config=None):
    if config is None:
        config = {}
    if not isinstance(config, dict) or set(config)-set(DEFAULT_OUTCOME):
        raise ValueError(f"outcome supports only {sorted(DEFAULT_OUTCOME)}")
    result = {key: _number(value, key) for key, value in {**DEFAULT_OUTCOME, **config}.items()}
    if not 0 < result["min_xy_overlap_fraction"] <= 1:
        raise ValueError("min_xy_overlap_fraction must be in (0,1]")
    for key in ("max_tilt_deg", "max_support_normal_tilt_deg"):
        if result[key] >= 90:
            raise ValueError(f"{key} must be below 90 degrees")
    return result


def _array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _cross(a, b):
    return float(a[0]*b[1]-a[1]*b[0])


def polygon_area(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 0.
    return abs(float(np.dot(points[:, 0], np.roll(points[:, 1], -1))-
                     np.dot(points[:, 1], np.roll(points[:, 0], -1))))/2


def intersection_area(subject, clip):
    """Convex CCW polygon clipping; avoids optional geometry dependencies."""
    output = [np.asarray(p, dtype=float) for p in subject]
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        if not output:
            return 0.
        old, output = output, []
        previous = old[-1]
        dp = _cross(b-a, previous-a)
        for current in old:
            dc = _cross(b-a, current-a)
            if (dc >= -1e-12) != (dp >= -1e-12):
                output.append(previous + dp/(dp-dc)*(current-previous))
            if dc >= -1e-12:
                output.append(current)
            previous, dp = current, dc
    return polygon_area(output)


def footprint(center_xy, edge_m, yaw_deg):
    angle = math.radians(float(yaw_deg))
    matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])*(float(edge_m)/2) @ matrix.T + center_xy


class CubeStackOutcome:
    """Success requires grasp/lift history and uninterrupted released support.

    A tick with BOTH fingers gripping the already lifted red cube is required;
    two unrelated historical touches do not establish a grasp. Support contacts
    must be filtered to near-vertical, near-surface contacts by the simulator.
    """
    def __init__(self, red_edge_m, blue_edge_m, tabletop_z, **config):
        self.edges = np.array([_number(red_edge_m, "red_edge_m"), _number(blue_edge_m, "blue_edge_m")])
        if isinstance(tabletop_z, (bool, np.bool_)) or not math.isfinite(float(tabletop_z)):
            raise ValueError("tabletop_z must be finite")
        self.tabletop_z = float(tabletop_z)
        self.config = outcome_config(config)
        self.grasped_and_lifted = False
        self.max_lift_m = 0.
        self.stable_since = None
        self.previous_seconds = None
        self.success = False
        self.first_success_time_s = None

    def update(self, *, seconds, positions, quaternions_wxyz, velocities, gripper_width_m, contacts):
        seconds = _number(seconds, "seconds", zero=True)
        if self.previous_seconds is not None and seconds <= self.previous_seconds:
            raise ValueError("Episode seconds must increase strictly")
        positions = _array(positions, (2, 3), "positions")
        quaternions = _array(quaternions_wxyz, (2, 4), "quaternions_wxyz")
        velocities = _array(velocities, (2, 6), "velocities")
        if np.any(np.abs(np.linalg.norm(quaternions, axis=1)-1.) > .01):
            raise ValueError("quaternions_wxyz must be normalized")
        width = _number(gripper_width_m, "gripper_width_m", zero=True)
        if not isinstance(contacts, dict):
            raise ValueError("contacts must be a mapping")
        known = {"red_finger_forces_n", "blue_finger_forces_n", "red_blue_support_force_n",
                 "blue_table_support_force_n", "red_table_force_n"}
        if set(contacts)-known:
            raise ValueError("Unknown cube contact fields")
        forces = {}
        for color in COLORS:
            key = color+"_finger_forces_n"
            forces[key] = _array(contacts.get(key, [0., 0.]), (2,), key)
            if np.any(forces[key] < 0):
                raise ValueError("Finger force magnitudes must be nonnegative")
        for key in known-{"red_finger_forces_n", "blue_finger_forces_n"}:
            forces[key] = _number(contacts.get(key, 0.), key, zero=True)
        self.previous_seconds = seconds
        cfg = self.config
        matrices = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()
        extents_z = np.sum(np.abs(matrices[:, 2, :]), axis=1)*self.edges/2
        bottoms, tops = positions[:, 2]-extents_z, positions[:, 2]+extents_z
        self.max_lift_m = max(self.max_lift_m, float(bottoms[0]-self.tabletop_z))
        bilateral = bool(np.all(forces["red_finger_forces_n"] >= cfg["min_grasp_force_n"]))
        lifted = float(bottoms[0]-self.tabletop_z) >= cfg["min_lift_height_m"]
        if (bilateral and lifted and width <= self.edges[0]+cfg["gripper_release_margin_m"]
                and forces["red_blue_support_force_n"] < cfg["min_support_force_n"]
                and forces["red_table_force_n"] < cfg["min_support_force_n"]):
            self.grasped_and_lifted = True
        upright = bool(np.all(matrices[:, 2, 2] >= math.cos(math.radians(cfg["max_tilt_deg"]))))
        square = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        faces = [square*(self.edges[i]/2) @ matrices[i, :2, :2].T + positions[i, :2] for i in range(2)]
        overlap = intersection_area(faces[0], faces[1])/max(polygon_area(faces[0]), 1e-12) if upright else 0.
        gap = float(bottoms[0]-tops[1])
        released = bool(width >= self.edges[0]+cfg["gripper_release_margin_m"] and
                        all(np.all(forces[color+"_finger_forces_n"] < cfg["min_grasp_force_n"]) for color in COLORS))
        conditions = dict(grasped_and_lifted=bool(self.grasped_and_lifted), upright=upright,
            sufficient_xy_overlap=bool(overlap >= cfg["min_xy_overlap_fraction"]),
            red_above_blue=bool(positions[0, 2] > positions[1, 2] and abs(gap) <= cfg["support_height_tolerance_m"]),
            red_blue_contact=bool(forces["red_blue_support_force_n"] >= cfg["min_support_force_n"]),
            blue_on_table=bool(abs(float(bottoms[1]-self.tabletop_z)) <= cfg["table_height_tolerance_m"] and
                               forces["blue_table_support_force_n"] >= cfg["min_support_force_n"]),
            red_not_table_supported=bool(forces["red_table_force_n"] < cfg["min_support_force_n"]),
            released=released,
            low_velocity=bool(np.all(np.linalg.norm(velocities[:, :3], axis=1) <= cfg["max_linear_speed_m_s"]) and
                              np.all(np.linalg.norm(velocities[:, 3:], axis=1) <= cfg["max_angular_speed_rad_s"])))
        valid = all(conditions.values())
        if not valid:
            self.stable_since = None
        elif self.stable_since is None:
            self.stable_since = seconds
        duration = 0. if self.stable_since is None else seconds-self.stable_since
        dropped = bool(np.any(tops < self.tabletop_z-cfg["drop_below_table_m"]))
        event = bool(valid and duration+1e-12 >= cfg["stable_seconds"] and not self.success and not dropped)
        if event:
            self.success, self.first_success_time_s = True, seconds
        return dict(success=bool(self.success and valid and duration+1e-12 >= cfg["stable_seconds"] and not dropped),
            ever_success=bool(self.success), success_event=event,
            grasped_and_lifted=bool(self.grasped_and_lifted), stable_seconds=float(duration),
            conditions=conditions, dropped=dropped, max_lift_m=float(self.max_lift_m),
            red_bottom_world_m=float(bottoms[0]), blue_top_world_m=float(tops[1]), blue_bottom_world_m=float(bottoms[1]),
            support_gap_m=gap, xy_overlap_fraction=float(overlap), first_success_time_s=self.first_success_time_s)
