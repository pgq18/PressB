"""Deterministic world-XY panel layouts with CPU safety checks.

The wall normal is world X; its complete back assembly follows fore-aft X
displacement, and the panel additionally slides horizontally along world Y.
The robot, home posture, table and both camera mounts stay fixed. Frustum
containment prevents cropping; it does not promise the robot cannot temporarily
occlude a button while pressing.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from itertools import product
import json

import numpy as np

from .planning import make_plan, panel_face_x


DEFAULT_PANEL_RANDOMIZATION = {
    "enabled": False,
    "min_offset_x_m": -.01,
    "max_offset_x_m": .01,
    "min_offset_y_m": -.025,
    "max_offset_y_m": .025,
    "camera_margin_px": 24.,
    "max_attempts": 8,
    "sampling_mode": "uniform",
    "grid_shape": [10, 10],
    "grid_seed": 0,
    "boundary_mode": "jitter",
}


def panel_randomization_settings(cfg):
    """Validate the serialized settings before planning or creating a scene."""
    settings = dict(DEFAULT_PANEL_RANDOMIZATION)
    supplied = cfg.get("panel_randomization", {})
    if not isinstance(supplied, dict):
        raise ValueError("panel_randomization must be an object")
    unknown = set(supplied) - set(settings)
    if unknown:
        raise ValueError(f"Unknown panel_randomization settings: {sorted(unknown)}")
    settings.update(supplied)
    if not isinstance(settings["enabled"], (bool, np.bool_)):
        raise ValueError("panel_randomization.enabled must be boolean")
    for name in ("min_offset_x_m", "max_offset_x_m", "min_offset_y_m", "max_offset_y_m", "camera_margin_px"):
        value = settings[name]
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"panel_randomization.{name} must be finite")
        try:
            settings[name] = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"panel_randomization.{name} must be finite") from error
        if not np.isfinite(settings[name]):
            raise ValueError(f"panel_randomization.{name} must be finite")
    for axis in ("x", "y"):
        if settings[f"min_offset_{axis}_m"] > settings[f"max_offset_{axis}_m"]:
            raise ValueError(f"panel_randomization minimum {axis} offset must not exceed maximum")
    if settings["camera_margin_px"] < 0:
        raise ValueError("panel_randomization.camera_margin_px must be nonnegative")
    attempts = settings["max_attempts"]
    if (isinstance(attempts, (bool, np.bool_)) or not isinstance(attempts, (int, np.integer))
            or not 1 <= attempts <= 128):
        raise ValueError("panel_randomization.max_attempts must be an integer from 1 to 128")
    settings["enabled"] = bool(settings["enabled"])
    settings["max_attempts"] = int(attempts)
    if settings["sampling_mode"] not in ("uniform", "stratified_grid"):
        raise ValueError("panel_randomization.sampling_mode must be uniform or stratified_grid")
    shape = settings["grid_shape"]
    if (not isinstance(shape, (list, tuple)) or len(shape) != 2
            or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1 for n in shape)
            or int(shape[0]) * int(shape[1]) > 10000):
        raise ValueError("panel_randomization.grid_shape must contain two positive integers with at most 10000 cells")
    settings["grid_shape"] = list(map(int, shape))
    grid_seed = settings["grid_seed"]
    if isinstance(grid_seed, (bool, np.bool_)) or not isinstance(grid_seed, (int, np.integer)) or grid_seed < 0:
        raise ValueError("panel_randomization.grid_seed must be a nonnegative integer")
    settings["grid_seed"] = int(grid_seed)
    if settings["boundary_mode"] not in ("jitter", "corners"):
        raise ValueError("panel_randomization.boundary_mode must be jitter or corners")
    if settings["enabled"] and settings["sampling_mode"] == "stratified_grid":
        if any(settings[f"min_offset_{axis}_m"] >= settings[f"max_offset_{axis}_m"] for axis in "xy"):
            raise ValueError("Stratified panel ranges must have positive width on both axes")
        if settings["boundary_mode"] == "corners" and min(shape) < 2:
            raise ValueError("Corner coverage needs at least two cells per axis")
    return settings


def _grid_cell(settings, floor, episode_index):
    """Assign one floor-local episode to its immutable shuffled stratum."""
    nx, ny = settings["grid_shape"]
    if (isinstance(episode_index, (bool, np.bool_)) or not isinstance(episode_index, (int, np.integer))
            or not 0 <= episode_index < nx * ny):
        raise ValueError(f"Stratified sampling requires episode_index in [0, {nx * ny - 1}] for each floor")
    if (isinstance(floor, (bool, np.bool_)) or not isinstance(floor, (int, np.integer)) or not 24 <= floor <= 35):
        raise ValueError("floor must be an integer from 24 through 35")
    stream = [settings["grid_seed"], int(floor), 0x47524944]
    rng = np.random.default_rng(np.random.SeedSequence(stream))
    flat_index = int(rng.permutation(nx * ny)[int(episode_index)])
    ix, iy = divmod(flat_index, ny)
    bounds = {}
    for axis, index, size in zip("xy", (ix, iy), (nx, ny)):
        edges = np.linspace(settings[f"min_offset_{axis}_m"], settings[f"max_offset_{axis}_m"], size + 1)
        bounds[axis] = [float(edges[index]), float(edges[index + 1])]
    return {"sampling_mode": "stratified_grid", "grid_shape": [nx, ny],
            "grid_seed": settings["grid_seed"], "grid_permutation_seed_stream": stream,
            "grid_permutation_order": "flat_x_major_y_minor", "episode_index": int(episode_index),
            "grid_flat_index": flat_index, "grid_cell_index_xy": [ix, iy],
            "grid_cell_bounds_m": bounds, "boundary_mode": settings["boundary_mode"],
            "boundary_corner": bool(settings["boundary_mode"] == "corners"
                                    and ix in (0, nx - 1) and iy in (0, ny - 1))}


def panel_camera_coverage(cfg, margin_px=24.):
    """Project the complete panel envelope using fixed_camera.py's D435 optics.

    The box encloses frame, faceplate, labels, screws, rings and button caps.
    Its eight corners suffice for a convex box completely ahead of the camera.
    Pixel margins refer to actual configured capture resolution.
    """
    offset = float(cfg.get("panel_offset_y_m", 0.))
    face_x = panel_face_x(cfg)
    margin_px = float(margin_px)
    if not np.isfinite(offset) or not np.isfinite(margin_px) or margin_px < 0:
        raise ValueError("Panel offset and camera margin must be finite; margin must be nonnegative")
    eye = np.asarray(cfg["global_camera_eye"], dtype=float)
    target = np.asarray(cfg["global_camera_target"], dtype=float)
    resolution = np.asarray(cfg.get("global_camera_resolution", [640, 480]), dtype=float)
    if (eye.shape != (3,) or target.shape != (3,) or not np.isfinite(eye).all()
            or not np.isfinite(target).all()):
        raise ValueError("Fixed camera eye/target must contain finite XYZ vectors")
    if (resolution.shape != (2,) or not np.isfinite(resolution).all()
            or np.any(resolution < 1) or np.any(resolution != np.floor(resolution))):
        raise ValueError("Fixed camera resolution must contain two positive integers")
    width, height = map(int, resolution)
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError("Fixed camera eye and target must differ")
    forward /= norm
    right = np.cross(forward, [0., 0., 1.])
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("Fixed camera direction cannot be parallel to world up")
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    focal = 1.88
    native_h = 2 * focal * np.tan(np.radians(69 / 2))
    native_v = 2 * focal * np.tan(np.radians(42 / 2))
    horizontal = min(native_h, native_v * width / height)
    vertical = horizontal * height / width
    fx, fy = focal / horizontal * width, focal / vertical * height
    mid_z = cfg["button_bottom_z"] + 2.5 * cfg["button_pitch_z"]
    half_height = (5 * cfg["button_pitch_z"] + .14) / 2
    points = np.array(list(product(
        # The raised digits/border project up to .225 mm ahead of the cap.
        [face_x - .0003, face_x + .049],
        [offset - .10, offset + .10],
        [mid_z - half_height, mid_z + half_height],
    )))
    relative = points - eye
    depth = relative @ forward
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.column_stack((fx * (relative @ right) / depth + width / 2,
                              height / 2 - fy * (relative @ up) / depth))
    margins = np.column_stack((uv[:, 0], width - 1 - uv[:, 0],
                               uv[:, 1], height - 1 - uv[:, 1]))
    inside = (np.isfinite(uv).all() and np.isfinite(depth).all()
              and bool(np.all((depth > .005) & (depth < 10.)))
              and bool(np.all(margins >= margin_px)))
    proof = {
        "all_inside": bool(inside), "points": len(points),
        "required_margin_px": margin_px, "minimum_margin_px": float(margins.min()),
        "minimum_optical_depth_m": float(depth.min()),
        "maximum_optical_depth_m": float(depth.max()),
        "pixel_bounds": [uv.min(0).tolist(), uv.max(0).tolist()],
        "resolution": [width, height], "intrinsics": [[fx, 0., width / 2], [0., fy, height / 2], [0., 0., 1.]],
        "panel_bounds_world_m": [points.min(0).tolist(), points.max(0).tolist()],
        "scope": "Complete panel envelope inside fixed D435 frustum; robot occlusion is not evaluated",
    }
    if not inside:
        raise ValueError(f"Complete panel fails fixed-camera coverage: minimum margin "
                         f"{proof['minimum_margin_px']:.3f} px, required {margin_px:.3f} px")
    return proof


def validate_panel_layout(kin, cfg, margin_px=24.):
    """Check all 12 full home/press/home paths, URDF limits and static clearance."""
    # Cache only small proof dictionaries, never trajectory/image arrays. A
    # repeated seed (preview/retry) can reuse its proof; distinct continuous
    # samples still receive independent complete checks.
    return deepcopy(_validate_panel_layout_cached(kin, json.dumps(cfg, sort_keys=True), float(margin_px)))


@lru_cache(maxsize=64)
def _validate_panel_layout_cached(kin, serialized_cfg, margin_px):
    # Delayed import avoids the episode generator's import cycle.
    from .dataset_planning import validate_episode_plan

    cfg = json.loads(serialized_cfg)
    coverage = panel_camera_coverage(cfg, margin_px)
    layout_cfg = dict(cfg, sequence=list(range(24, 36)))
    plan = make_plan(kin, layout_cfg)
    checks = validate_episode_plan(kin, plan, layout_cfg)
    minimum_joint_margin = float(np.minimum(plan.q - kin.lower, kin.upper - plan.q).min())
    return {"panel_offset_y_m": float(cfg.get("panel_offset_y_m", 0.)),
            "panel_offset_x_m": float(cfg.get("panel_offset_x_m", 0.)),
            "reachable_floors": list(range(24, 36)), "all_buttons_reachable": True,
            "trajectory_scope": "All 12 nominal home/approach/press/retract/home trajectories",
            "camera_coverage": coverage, "trajectory_validation": checks,
            "minimum_joint_limit_margin_rad": minimum_joint_margin}


def sample_panel_layout(kin, cfg, floor, seed, *, episode_index=None):
    """Return an episode config and replayable sampling/safety metadata.

    Offset bounds are absolute displacement from the original centered panel,
    not increments added to the preceding episode. A separate seeded stream
    preserves the historical approach/timing draws exactly when disabled.
    Stratified mode needs the floor-local episode_index; worker execution order
    never enters the cell assignment. Rejection samples stay inside that cell.
    """
    settings = panel_randomization_settings(cfg)
    episode_cfg = dict(cfg)
    existing_offset = float(cfg.get("panel_offset_y_m", 0.))
    existing_offset_x = float(cfg.get("panel_offset_x_m", 0.))
    if not np.isfinite(existing_offset) or not np.isfinite(existing_offset_x):
        raise ValueError("panel_offset_x_m and panel_offset_y_m must be finite")
    if not settings["enabled"]:
        proof = {"enabled": False, "panel_offset_y_m": existing_offset, "panel_offset_x_m": existing_offset_x}
        if existing_offset != 0 or existing_offset_x != 0:
            proof["validation"] = validate_panel_layout(kin, episode_cfg, settings["camera_margin_px"])
        return episode_cfg, proof
    # Seed word encodes "PANE"; floor and seed alone identify any sampled layout.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(floor), 0x50414E45]))
    grid = (_grid_cell(settings, floor, episode_index)
            if settings["sampling_mode"] == "stratified_grid" else None)
    rejected = []
    for attempt in range(1, settings["max_attempts"] + 1):
        if grid is None:
            # Keep the established draws and metadata byte-for-byte compatible
            # with the original independently uniform layout sampler.
            offset_x = float(rng.uniform(settings["min_offset_x_m"], settings["max_offset_x_m"]))
            offset = float(rng.uniform(settings["min_offset_y_m"], settings["max_offset_y_m"]))
        elif grid["boundary_corner"]:
            offset_x, offset = (settings[f"{'min' if index == 0 else 'max'}_offset_{axis}_m"]
                                for axis, index in zip("xy", grid["grid_cell_index_xy"]))
        else:
            # Half-open cells make independent floor/bin audits unambiguous;
            # nextafter only handles floating-point rounding onto the high edge.
            offset_x, offset = (float(min(rng.uniform(low, high), np.nextafter(high, low)))
                                for low, high in (grid["grid_cell_bounds_m"][axis] for axis in "xy"))
        episode_cfg["panel_offset_x_m"] = offset_x
        episode_cfg["panel_offset_y_m"] = offset
        try:
            validation = validate_panel_layout(kin, episode_cfg, settings["camera_margin_px"])
        except (ValueError, RuntimeError) as error:
            rejected.append({"panel_offset_y_m": offset, "panel_offset_x_m": offset_x, "reason": str(error)})
            if grid is not None and grid["boundary_corner"]:
                raise ValueError(f"Required panel corner {grid['grid_cell_index_xy']} is unsafe: {error}") from error
            continue
        proof = {"enabled": True, "axis": "world_xy", "distribution": "uniform_rectangle_with_safety_rejection",
                             "seed_stream": [int(seed), int(floor), 0x50414E45],
                             "min_offset_y_m": settings["min_offset_y_m"],
                             "max_offset_y_m": settings["max_offset_y_m"],
                             "min_offset_x_m": settings["min_offset_x_m"],
                             "max_offset_x_m": settings["max_offset_x_m"],
                             "panel_offset_x_m": offset_x,
                             "panel_offset_y_m": offset, "attempts": attempt,
                             "rejected_candidates": rejected, "validation": validation}
        if grid is not None:
            proof.update(grid, distribution="stratified_grid_with_safety_rejection")
        return episode_cfg, proof
    raise ValueError(f"No safe panel layout after {settings['max_attempts']} attempts; "
                     f"last rejection: {rejected[-1]['reason']}")
