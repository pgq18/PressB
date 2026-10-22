"""Language, placement and measured press outcomes for the tabletop task.

All positions are in metres relative to the PiPER base. Success is derived from
the moving cap and its contacts with the gripper fingers, never a commanded TCP
position. This module has no Isaac dependency.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np


COLORS = ("red", "yellow", "blue")
_CHINESE_COLORS = {"红": "red", "黄": "yellow", "蓝": "blue"}


def task_instruction(color: str) -> str:
    if color not in COLORS:
        raise ValueError(f"Button color must be one of {COLORS}")
    return f"Press the {color} button."


def parse_instruction(instruction: str) -> str:
    """Accept explicit single-button English/Chinese commands; reject ambiguity.

    Deliberately use a whole-command match: a negated instruction or a command
    naming two buttons must not silently choose whichever color appears first.
    """
    if not isinstance(instruction, str):
        raise ValueError("Task instruction must be text")
    text = re.sub(r"\s+", " ", instruction.strip()).rstrip(".!。！").strip()
    english = re.fullmatch(r"(?:please )?press (?:the )?(red|yellow|blue) button", text, re.I)
    if english:
        return english.group(1).lower()
    chinese = re.fullmatch(r"请?按(?:下)?\s*([红黄蓝])色?\s*(?:按钮|按键)", text)
    if chinese:
        return _CHINESE_COLORS[chinese.group(1)]
    raise ValueError("Expected a single instruction such as 'Press the red button.' or '按下红色按键'")


def project_base_points(points, up_camera):
    """Project base-frame 3D points using the PAP camera's USD transform.

    USD cameras look down local -Z, with +Y pointing up. Returned depth is
    positive in front of the camera; image v points down.
    """
    points = np.asarray(points, dtype=float)
    camera = np.asarray(up_camera["base_from_camera_usd"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx3 base-frame points")
    if camera.shape != (4, 4) or not np.isfinite(camera).all():
        raise ValueError("Expected a finite 4x4 camera transform")
    local = (np.linalg.inv(camera) @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    depth = -local[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = np.column_stack((
            float(up_camera["cx"]) + float(up_camera["fx"]) * local[:, 0] / depth,
            float(up_camera["cy"]) - float(up_camera["fy"]) * local[:, 1] / depth,
        ))
    return pixels, depth


def button_in_camera(center_xy, up_camera, *, button_radius=.022,
                     button_top_z=.025, margin_px=12.) -> bool:
    """Require the full housing's enclosing box inside the third-person image."""
    xy = np.asarray(center_xy, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        return False
    if not np.isfinite([button_radius, button_top_z, margin_px]).all() or button_radius <= 0 or button_top_z <= 0 or margin_px < 0:
        raise ValueError("Invalid button dimensions or image margin")
    corners = [[xy[0]+dx, xy[1]+dy, z]
               for dx in (-button_radius, button_radius)
               for dy in (-button_radius, button_radius)
               for z in (0., button_top_z)]
    pixels, depth = project_base_points(corners, up_camera)
    resolution = np.asarray(up_camera["resolution"], dtype=float)
    return bool(np.all(depth > 0) and np.isfinite(pixels).all() and
                np.all(pixels >= margin_px) and
                np.all(pixels <= resolution - 1 - margin_px))


def sample_button_positions(up_camera, seed=None, *,
                            bounds_xy=((.30, .39), (-.14, .14)),
                            button_radius=.022, min_separation=.10,
                            button_top_z=.025, margin_px=12., max_attempts=3000):
    """Sample three separated, visible positions with a reproducible seed.

    The default rectangle is intended for the demo's radially oriented gripper
    pitched 160 degrees about Y, with z=.12 m approach height. Visibility is
    checked against the inherited camera, including the housing perimeter.
    Custom bounds/approach orientations need their own reachability check.
    """
    bounds = np.asarray(bounds_xy, dtype=float)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("bounds_xy must contain increasing finite x/y intervals")
    if not np.isfinite(min_separation) or min_separation < 2 * button_radius:
        raise ValueError("Minimum separation must leave button housings disjoint")
    if not isinstance(max_attempts, int) or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    rng = np.random.default_rng(seed)
    # Restart partial layouts rather than getting stuck behind one unlucky
    # centre placement when the requested separation is near workspace size.
    for attempt in range(max_attempts):
        if attempt % 100 == 0:
            positions = {}
        point = rng.uniform(bounds[:, 0], bounds[:, 1])
        if not button_in_camera(point, up_camera, button_radius=button_radius,
                                button_top_z=button_top_z, margin_px=margin_px):
            continue
        if any(np.linalg.norm(point-np.asarray(other)) < min_separation for other in positions.values()):
            continue
        positions[COLORS[len(positions)]] = point.tolist()
        if len(positions) == len(COLORS):
            return positions
    raise ValueError("Cannot place three separated buttons inside the requested camera/workspace bounds")


class TableButtonOutcome:
    """Debounce physical presses, with wrong-color presses ending the episode.

    A press needs simultaneous cap displacement and measured finger force for
    the complete dwell period. Hysteresis rearms a cap only after it springs
    back below release_depth_m. A simultaneous target/wrong press is a failure.
    Terminal outcome is latched until a new checker is constructed for reset.
    """

    def __init__(self, target_color, *, press_depth_m=.003, release_depth_m=.001,
                 min_contact_force_n=.1, hold_seconds=.025):
        task_instruction(target_color)
        values = [press_depth_m, release_depth_m, min_contact_force_n, hold_seconds]
        if (not np.isfinite(values).all() or not 0 <= release_depth_m < press_depth_m
                or min_contact_force_n <= 0 or hold_seconds < 0):
            raise ValueError("Invalid press depth, release hysteresis, contact force, or hold duration")
        self.target_color = target_color
        self.press_depth_m = float(press_depth_m)
        self.release_depth_m = float(release_depth_m)
        self.min_contact_force_n = float(min_contact_force_n)
        self.hold_seconds = float(hold_seconds)
        self._since = {color: None for color in COLORS}
        self._latched = {color: False for color in COLORS}
        self._last_seconds = None
        self.success = False
        self.wrong_button = None
        self.first_success_time_s = None
        self.failure_time_s = None

    def update(self, seconds, travels: Mapping, contact_forces: Mapping):
        """Read positive-down cap travel (m) and finger-contact force magnitude (N)."""
        seconds = float(seconds)
        if not np.isfinite(seconds) or (self._last_seconds is not None and seconds < self._last_seconds):
            raise ValueError("Physical sample timestamps must be finite and nondecreasing")
        if set(travels) != set(COLORS) or set(contact_forces) != set(COLORS):
            raise ValueError("Supply cap travel and finger contact force for all three colors")
        depth = {c: float(travels[c]) for c in COLORS}
        force = {c: float(contact_forces[c]) for c in COLORS}
        if not np.isfinite(list(depth.values()) + list(force.values())).all() or min(force.values()) < 0:
            raise ValueError("Cap travel and nonnegative contact forces must be finite")
        self._last_seconds = seconds
        events = []
        for color in COLORS:
            if depth[color] <= self.release_depth_m:
                self._latched[color] = False
            valid = depth[color] >= self.press_depth_m and force[color] >= self.min_contact_force_n
            if not valid or self._latched[color]:
                self._since[color] = None
                continue
            if self._since[color] is None:
                self._since[color] = seconds
            if seconds - self._since[color] + 1e-12 >= self.hold_seconds:
                self._latched[color] = True
                self._since[color] = None
                events.append(color)
        if not self.success and self.wrong_button is None:
            wrong = [color for color in events if color != self.target_color]
            if wrong:
                self.wrong_button = wrong[0]
                self.failure_time_s = seconds
            elif self.target_color in events:
                self.success = True
                self.first_success_time_s = seconds
        return dict(target_color=self.target_color, success=self.success,
                    wrong_button=self.wrong_button,
                    terminated=bool(self.success or self.wrong_button is not None),
                    reward=float(self.success), press_events=events,
                    pressed_colors=[c for c in COLORS if self._latched[c]],
                    travel_m=depth, finger_contact_force_n=force,
                    first_success_time_s=self.first_success_time_s,
                    failure_time_s=self.failure_time_s)
