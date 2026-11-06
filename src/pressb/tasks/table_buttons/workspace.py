"""URDF-based press workspace, reusing the PiPER's exact first-axis symmetry.

The lookup profiles are a fast proposal filter, not a reachability assertion:
``solve`` verifies the requested radius and complete sampled approach, then
rotates its actual joint solutions to the requested XY. Camera visibility,
robot/environment collision checks and transit planning belong to the caller.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class PressWorkspace:
    """Find a connected hover-to-press branch, preferring the original 160° tilt.

    ``cfg['press_workspace']`` optionally controls the documented defaults below.
    Profiles are computed lazily and retained on the instance. The radius-only
    reuse is valid because the bundled URDF's first joint rotates about base Z
    on the base Z axis; this property is checked instead of merely assumed.
    """

    def __init__(self, kin, cfg):
        self.kin = kin
        self.initial = np.asarray(cfg["initial_joint_positions_rad"], dtype=float)
        if self.initial.shape != (6,) or not np.isfinite(self.initial).all():
            raise ValueError("Expected six finite initial arm joints")
        options = cfg.get("press_workspace", {})
        supported = {"pitch_candidates_deg", "hover_z_m", "hover_z_candidates_m", "press_offset_m", "radial_step_m",
                     "radial_bounds_m", "vertical_samples", "max_joint_step_rad"}
        if not isinstance(options, dict) or set(options) - supported:
            raise ValueError(f"press_workspace supports only {sorted(supported)}")
        self.pitch_candidates_deg = tuple(float(x) for x in options.get("pitch_candidates_deg", [160, 150, 140, 130, 170]))
        self.hover_z_candidates_m = tuple(float(x) for x in options.get(
            "hover_z_candidates_m", [options["hover_z_m"]] if "hover_z_m" in options else [.12, .08]))
        geometry = cfg.get("button_geometry", {})
        self.top_z_m = float(geometry.get("base_height_m", .017)) + float(geometry.get("cap_height_m", .008))
        self.press_z_m = self.top_z_m - float(options.get("press_offset_m", .006))
        self.near_z_m = self.top_z_m + .018
        self.radial_step_m = float(options.get("radial_step_m", .01))
        self.radial_bounds_m = tuple(float(x) for x in options.get("radial_bounds_m", [.06, .70]))
        self.vertical_samples = options.get("vertical_samples", 9)
        self.max_joint_step_rad = float(options.get("max_joint_step_rad", .45))
        scalars = [*self.pitch_candidates_deg, *self.hover_z_candidates_m, self.top_z_m, self.press_z_m,
                   self.radial_step_m, *self.radial_bounds_m, self.max_joint_step_rad]
        if not np.isfinite(scalars).all() or not self.pitch_candidates_deg or not self.hover_z_candidates_m:
            raise ValueError("Workspace parameters must be finite with at least one pitch")
        if any(not 90 < pitch <= 180 for pitch in self.pitch_candidates_deg):
            raise ValueError("Press pitch candidates must be in (90, 180] degrees")
        if (len(self.radial_bounds_m) != 2 or not 0 < self.radial_bounds_m[0] < self.radial_bounds_m[1]
                or not 0 < self.radial_step_m <= .05 or not 0 <= self.press_z_m < self.near_z_m < min(self.hover_z_candidates_m)
                or type(self.vertical_samples) is not int or self.vertical_samples < 3 or self.max_joint_step_rad <= 0):
            raise ValueError("Invalid workspace bounds, spacing or approach heights")
        # A fixed transform may precede joint1; include it when checking the
        # rotation origin/axis, so a different robot cannot reuse this shortcut.
        first_frame = np.eye(4)
        for origin, axis in kin._chain:
            first_frame = first_frame @ origin
            if axis is not None:
                if not (np.allclose(first_frame[:2, 3], 0., atol=1e-10)
                        and np.allclose(first_frame[:3, :3] @ axis, [0., 0., 1.], atol=1e-10)):
                    raise ValueError("Radial reuse requires first joint on the base +Z axis")
                break
        self._profiles = None
        self._solutions = {}
        self._rotations = {pitch: Rotation.from_euler("y", pitch, degrees=True).as_matrix()
                           for pitch in self.pitch_candidates_deg}
        # Include the close-approach waypoint exactly, rather than merely
        # checking hover/press endpoints with an untested gap between them.
        self._heights = {hover: np.unique(np.r_[np.linspace(hover, self.press_z_m, self.vertical_samples), self.near_z_m])[::-1]
                         for hover in self.hover_z_candidates_m}

    def _radial_path(self, radius, pitch, hover, seed):
        key = (round(float(radius), 12), pitch, hover)
        if key in self._solutions:
            return self._solutions[key]
        orientation = self._rotations[pitch]
        joints = []
        try:
            for height in self._heights[hover]:
                q = self.kin.solve([radius, 0., height], seed, orientation)
                if joints and np.max(np.abs(q - joints[-1])) > self.max_joint_step_rad:
                    self._solutions[key] = None
                    return None
                joints.append(q)
                seed = q
        except ValueError:
            self._solutions[key] = None
            return None
        result = np.asarray(joints)
        self._solutions[key] = result
        return result

    def _build_profiles(self):
        if self._profiles is not None:
            return
        low, high = self.radial_bounds_m
        radii = np.arange(low, high + self.radial_step_m * .25, self.radial_step_m)
        profiles = []
        for hover in self.hover_z_candidates_m:
            for pitch in self.pitch_candidates_deg:
                seed = self.initial.copy()
                samples, intervals, interval = [], [], None
                for radius in radii:
                    path = self._radial_path(float(radius), pitch, hover, seed)
                    feasible = path is not None
                    samples.append(dict(radius_m=float(radius), feasible=feasible))
                    if feasible:
                        seed = path[0]
                        if interval is None:
                            interval = [float(radius), float(radius)]
                        interval[1] = float(radius)
                    elif interval is not None:
                        intervals.append(interval)
                        interval = None
                if interval is not None:
                    intervals.append(interval)
                profiles.append(dict(pitch_deg=pitch, hover_z_m=hover, press_z_m=self.press_z_m,
                                     intervals_m=intervals, samples=samples))
        self._profiles = profiles

    @property
    def radial_profiles(self):
        """JSON-compatible sampled radial intervals, excluding camera/collisions."""
        self._build_profiles()
        return self._profiles

    def solutions(self, xy):
        """Yield exact-XY approaches so callers can reject colliding candidates.

        Returned arrays: ``orientation``, ``hover``, ``near``, ``press``,
        ``q_hover``, ``q_press``, ``q_path``, ``path_xyz``. Scalar fields record
        the selected pitch, hover height, radius, yaw and sampled joint margin.
        """
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("Expected two finite button XY coordinates")
        radius = float(np.linalg.norm(xy))
        yaw = float(np.arctan2(xy[1], xy[0]))
        if (not self.radial_bounds_m[0] <= radius <= self.radial_bounds_m[1]
                or yaw < self.kin.lower[0] - .01 or yaw > self.kin.upper[0] + .01):
            return
        self._build_profiles()
        yaw_rotation = Rotation.from_euler("z", yaw).as_matrix()
        for profile in self._profiles:
            # Expand by one grid spacing, then do exact IK. This retains the
            # unknown part of each sampled interval boundary without declaring
            # those extra radii feasible merely because a neighbour passed.
            if not any(lo-self.radial_step_m <= radius <= hi+self.radial_step_m
                       for lo, hi in profile["intervals_m"]):
                continue
            pitch = profile["pitch_deg"]
            hover = profile["hover_z_m"]
            nearest = min((row for row in profile["samples"] if row["feasible"]),
                          key=lambda row: abs(row["radius_m"]-radius))
            seed = self._solutions[(round(nearest["radius_m"], 12), pitch, hover)][0]
            radial_path = self._radial_path(radius, pitch, hover, seed)
            if radial_path is None:
                continue
            q_path = radial_path.copy()
            q_path[:, 0] += yaw
            if np.any(q_path < self.kin.lower) or np.any(q_path > self.kin.upper):
                continue
            orientation = yaw_rotation @ self._rotations[pitch]
            path_xyz = np.c_[np.tile(xy, (len(q_path), 1)), self._heights[hover]]
            valid = True
            for q, xyz in zip(q_path, path_xyz):
                actual = self.kin.fk(q)
                if (np.linalg.norm(actual[:3, 3]-xyz) >= 5e-4 or
                        Rotation.from_matrix(orientation @ actual[:3, :3].T).magnitude() >= 1e-2):
                    valid = False
                    break
            if valid:
                margin = np.minimum(q_path-self.kin.lower, self.kin.upper-q_path)
                yield dict(pitch_deg=pitch, hover_z_m=hover, press_z_m=self.press_z_m,
                            radius_m=radius, yaw_rad=yaw, orientation=orientation,
                            hover=np.r_[xy, hover], near=np.r_[xy, self.near_z_m],
                            press=np.r_[xy, self.press_z_m], q_hover=q_path[0].copy(),
                            q_press=q_path[-1].copy(), q_path=q_path, path_xyz=path_xyz,
                            minimum_joint_limit_margin_rad=float(np.min(margin)),
                            vertical_sample_count=len(q_path))

    def solve(self, xy):
        """Return the first connected IK approach, or None when all fail."""
        return next(self.solutions(xy), None)
