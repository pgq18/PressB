"""Sample visible tabletop button centres with exact IK and path clearance.

The camera polygon supplies proposals, not robot feasibility. Every accepted
point is solved at its own XY and screened along the shared press trajectory.
``grid`` is a diagnostic view of these same checks, never a sampling lookup.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

import numpy as np

from .camera_domain import camera_center_domain
from .collision import RobotClearance
from .motion import screening_path
from .task import COLORS, button_in_camera
from .workspace import PressWorkspace


def _in_polygon(points, polygon):
    """Membership in the convex camera/table intersection, including edges."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(polygon) < 3:
        return np.zeros(len(points), dtype=bool)
    edges = np.roll(polygon, -1, axis=0) - polygon
    offsets = points[:, None, :] - polygon[None, :, :]
    cross = edges[None, :, 0] * offsets[..., 1] - edges[None, :, 1] * offsets[..., 0]
    return np.all(cross >= -1e-12, axis=1) | np.all(cross <= 1e-12, axis=1)


class ReachablePlacement:
    """Intersect the full camera domain with collision-screened press paths.

    Collision checks use the configured initial gripper width, which the demo
    keeps throughout the press. Paths are discretely checked by
    :func:`screening_path`; acceptance does not claim continuous collision
    detection or visibility through robot occlusion.
    """

    def __init__(self, kin, cfg):
        self.cfg = deepcopy(cfg)
        self.domain = camera_center_domain(self.cfg)
        self._polygon = np.asarray(self.domain["polygon_xy_m"], dtype=float).reshape(-1, 2)
        bounds = self.domain["bounds_xy_m"]
        self._bounds = None if bounds is None else np.asarray(bounds, dtype=float)
        self.initial_q = np.asarray(self.cfg["initial_joint_positions_rad"], dtype=float)
        self.gripper_width = float(self.cfg.get("initial_gripper_width_m", 0.))
        geometry = self.cfg.get("button_geometry", {})
        self.radius = float(geometry.get("base_radius_m", .022))
        self.top = float(geometry.get("base_height_m", .017)) + float(geometry.get("cap_height_m", .008))
        sampling = self.cfg.get("position_sampling", {})
        self.margin = float(sampling.get("pixel_margin", 16.))
        self.min_separation = float(sampling.get("min_separation", .10))
        self.max_attempts = sampling.get("max_attempts", 3000)
        if self.initial_q.shape != (6,) or not np.isfinite(self.initial_q).all():
            raise ValueError("Expected six finite initial arm joints")
        if not np.isfinite(self.gripper_width) or self.gripper_width < 0:
            raise ValueError("Initial gripper width must be finite and nonnegative")
        if not np.isfinite(self.min_separation) or self.min_separation < 2*self.radius:
            raise ValueError("Minimum separation must leave button housings disjoint")
        if type(self.max_attempts) is not int or self.max_attempts <= 0:
            raise ValueError("position_sampling.max_attempts must be a positive integer")
        self.workspace = PressWorkspace(kin, self.cfg)
        self.clearance = RobotClearance(kin, self.cfg)
        self.last_diagnostic = {"reason": "not_checked"}

    def _visible(self, xy):
        return bool(_in_polygon(np.asarray(xy)[None], self._polygon)[0] and button_in_camera(
            xy, self.cfg["up_camera"], button_radius=self.radius,
            button_top_z=self.top, margin_px=self.margin))

    def candidate(self, xy):
        """Return the first clear, exact-XY IK approach, or ``None``.

        All orientation/height candidates remain eligible after a collision in
        the preferred approach. ``last_diagnostic`` reports the latest rejection
        or the accepted path's collision-check metadata.
        """
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("Expected two finite button XY coordinates")
        if not self._visible(xy):
            self.last_diagnostic = {"reason": "outside_camera_domain"}
            return None
        rejected = []
        for approach in self.workspace.solutions(xy):
            path = screening_path(self.initial_q, approach)
            ok, diagnostic = self.clearance.check_path(path, self.gripper_width)
            if not ok:
                rejected.append(deepcopy(diagnostic))
                continue
            # A single candidate still has its own housing and cap. Only the
            # intended finger/cap contact is exempt from the obstacle check.
            button_ok, button_diagnostic = self.clearance.check_button_path(
                path, {COLORS[0]: xy.tolist()}, COLORS[0], self.gripper_width)
            if not button_ok:
                rejected.append(deepcopy(button_diagnostic))
                continue
            result = dict(approach)
            result["clearance"] = deepcopy(diagnostic)
            result["button_clearance"] = deepcopy(button_diagnostic)
            result["screening_sample_count"] = len(path)
            self.last_diagnostic = {"reason": "clear", "clearance": deepcopy(diagnostic),
                                    "rejected_approaches": len(rejected)}
            return result
        self.last_diagnostic = {"reason": "path_collision" if rejected else "no_ik",
                                "rejected_approaches": rejected}
        return None

    def validate_layout(self, positions, approaches):
        """Check every selected path against all three placed buttons.

        The target cap permits intended finger contact, while its housing and
        every non-target cap remain obstacles. This also serves callers using a
        fixed configuration instead of random sampling.
        """
        if set(positions) != set(COLORS) or set(approaches) != set(COLORS):
            raise ValueError("Supply positions and approaches for all three button colors")
        points = np.asarray([positions[color] for color in COLORS], dtype=float)
        if points.shape != (3, 2) or not np.isfinite(points).all():
            raise ValueError("Button positions must each contain two finite coordinates")
        for index, color in enumerate(COLORS):
            if not np.allclose(np.asarray(approaches[color]["press"])[:2], points[index], rtol=0., atol=1e-8):
                raise ValueError(f"Selected approach belongs to a different {color} button position")
            for other in range(index+1, len(COLORS)):
                if np.linalg.norm(points[index]-points[other]) < self.min_separation:
                    return False, {"reason": "button_separation", "pair": [color, COLORS[other]]}
        checks = {}
        for color in COLORS:
            path = screening_path(self.initial_q, approaches[color])
            ok, diagnostic = self.clearance.check_button_path(path, positions, color, self.gripper_width)
            checks[color] = deepcopy(diagnostic)
            if not ok:
                return False, {"reason": "button_path_collision", "target_color": color,
                               "checks": checks}
        return True, {"reason": "clear", "checks": checks}

    def sample(self, seed=None):
        """Uniformly propose centres over the entire camera-domain bounds.

        The polygon, exact IK, path clearance and pairwise spacing are applied
        by rejection sampling. Partial layouts restart after 100 proposals so
        an unfortunate first point cannot permanently exclude the other colors.
        """
        if self._bounds is None:
            raise ValueError("Cannot place buttons: the camera/table domain is empty")
        rng = np.random.default_rng(seed)
        rejected = Counter()
        positions, approaches = {}, {}
        restarts = 0
        for attempt in range(self.max_attempts):
            if attempt % 100 == 0:
                restarts += int(attempt > 0)
                positions, approaches = {}, {}
            point = rng.uniform(self._bounds[:, 0], self._bounds[:, 1])
            if any(np.linalg.norm(point-np.asarray(other)) < self.min_separation
                   for other in positions.values()):
                rejected["separation"] += 1
                continue
            approach = self.candidate(point)
            if approach is None:
                rejected[self.last_diagnostic["reason"]] += 1
                continue
            color = COLORS[len(positions)]
            positions[color], approaches[color] = point.tolist(), approach
            if len(positions) == len(COLORS):
                ok, diagnostic = self.validate_layout(positions, approaches)
                if not ok:
                    rejected[diagnostic["reason"]] += 1
                    positions, approaches = {}, {}
                    restarts += 1
                    continue
                domain = deepcopy(self.domain)
                domain["sampling"] = dict(
                    proposal="uniform over complete camera-domain bounds",
                    exact_candidate_ik=True, discrete_path_collision_check=True,
                    complete_button_layout_check=True,
                    min_separation_m=self.min_separation, attempts=attempt+1,
                    layout_restarts=restarts, rejected_by_reason=dict(rejected))
                domain["layout_clearance"] = diagnostic
                return {"positions": positions, "approaches": approaches, "domain": domain}
        raise ValueError(
            f"Cannot place three separated, reachable buttons after {self.max_attempts} proposals "
            f"in the camera/table domain; rejected={dict(rejected)}")

    def grid(self, step_m=.01):
        """Evaluate a diagnostic grid; every visible point gets exact IK.

        ``approaches`` and ``reasons`` are aligned with the flat ``points_xy``
        array. The reachable mask describes single-button approaches; it is not
        an assertion that arbitrary simultaneous layouts are mutually clear.
        """
        step_m = float(step_m)
        if not np.isfinite(step_m) or step_m <= 0:
            raise ValueError("Grid spacing must be positive and finite")
        if self._bounds is None:
            points = np.empty((0, 2), dtype=float)
        else:
            axes = [np.arange(low, high+step_m*1e-9, step_m) for low, high in self._bounds]
            x, y = np.meshgrid(*axes, indexing="xy")
            points = np.column_stack((x.ravel(), y.ravel()))
        visible = np.array([self._visible(point) for point in points], dtype=bool)
        reachable = np.zeros(len(points), dtype=bool)
        approaches, reasons = [None]*len(points), ["outside_camera_domain"]*len(points)
        for index in np.flatnonzero(visible):
            approach = self.candidate(points[index])
            approaches[index] = approach
            reasons[index] = self.last_diagnostic["reason"]
            reachable[index] = approach is not None
        return {"points_xy": points, "visible_mask": visible, "reachable_mask": reachable,
                "approaches": approaches, "reasons": reasons,
                "bounds_xy_m": deepcopy(self.domain["bounds_xy_m"]), "resolution_m": step_m,
                "domain": deepcopy(self.domain)}
