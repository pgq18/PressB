"""Recorded panel-layout checks shared by collectors, exports and audits.

Offsets describe an absolute translation along environment-local world X/Y.
They are episode context, never a policy observation or action dimension.
Legacy captures without these fields retain their original zero offset.
"""
from __future__ import annotations

import math


PANEL_EPISODE_CONTEXT = ("panel_offset_x_m", "panel_offset_y_m", "panel_layout")


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Invalid finite panel metadata number: {name}")
    return float(value)


def _same(left, right, name):
    if abs(_number(left, name) - _number(right, name)) > 1e-12:
        raise ValueError(f"Panel offset differs between recorded contexts: {name}")


def _finite_tree(value, name):
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{name}[{index}]")
    elif type(value) is float and not math.isfinite(value):
        raise ValueError(f"Nonfinite panel evidence: {name}")


def episode_panel_context(metadata):
    """Copy only recorded fields, preserving old manifest identities exactly."""
    return {key: metadata[key] for key in PANEL_EPISODE_CONTEXT if key in metadata}


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid panel metadata integer: {name}")
    return value


def _validate_stratified_context(randomization, metadata, proof, offsets):
    """Check one stratum geometrically; whole-floor occupancy is audited separately."""
    if (proof.get("sampling_mode") != "stratified_grid"
            or proof.get("distribution") != "stratified_grid_with_safety_rejection"):
        raise ValueError("Missing or inconsistent stratified panel sampling mode")
    shape = randomization.get("grid_shape", [10, 10])
    if (not isinstance(shape, (list, tuple)) or len(shape) != 2
            or any(type(value) is not int or value < 1 for value in shape)
            or shape[0] * shape[1] > 10000 or proof.get("grid_shape") != list(shape)):
        raise ValueError("Invalid or inconsistent panel grid shape")
    grid_seed = _integer(randomization.get("grid_seed", 0), "configured grid_seed")
    if _integer(proof.get("grid_seed"), "grid_seed") != grid_seed:
        raise ValueError("Panel grid seed differs from configuration")
    floor = _integer(metadata.get("floor"), "floor", 24)
    episode_id = _integer(metadata.get("episode_id"), "episode_id")
    index = _integer(proof.get("episode_index"), "episode_index")
    if floor > 35 or floor != 24 + episode_id % 12 or index != episode_id // 12 or index >= shape[0] * shape[1]:
        raise ValueError("Panel grid episode index differs from floor-local episode identity")
    if (proof.get("grid_permutation_seed_stream") != [grid_seed, floor, 0x47524944]
            or proof.get("grid_permutation_order") != "flat_x_major_y_minor"):
        raise ValueError("Panel grid permutation identity differs from configuration")
    if proof.get("seed_stream") != [_integer(metadata.get("seed"), "seed"), floor, 0x50414E45]:
        raise ValueError("Panel jitter seed stream differs from episode identity")
    cell = proof.get("grid_cell_index_xy")
    if (not isinstance(cell, list) or len(cell) != 2
            or any(type(value) is not int or not 0 <= value < size for value, size in zip(cell, shape))
            or _integer(proof.get("grid_flat_index"), "grid_flat_index") != cell[0] * shape[1] + cell[1]):
        raise ValueError("Invalid panel grid cell index")
    boundary_mode = randomization.get("boundary_mode", "jitter")
    if boundary_mode not in ("jitter", "corners") or proof.get("boundary_mode") != boundary_mode:
        raise ValueError("Panel grid boundary mode differs from configuration")
    if boundary_mode == "corners" and min(shape) < 2:
        raise ValueError("Corner coverage needs at least two grid cells per axis")
    corner = boundary_mode == "corners" and all(i in (0, n - 1) for i, n in zip(cell, shape))
    if proof.get("boundary_corner") is not corner:
        raise ValueError("Panel grid corner marker differs from its cell")
    bounds = proof.get("grid_cell_bounds_m")
    if not isinstance(bounds, dict) or set(bounds) != {"x", "y"}:
        raise ValueError("Missing panel grid cell bounds")
    expected_bounds = {}
    for axis, cell_index, size in zip("xy", cell, shape):
        low = _number(randomization.get(f"min_offset_{axis}_m"), f"min_offset_{axis}_m")
        high = _number(randomization.get(f"max_offset_{axis}_m"), f"max_offset_{axis}_m")
        if low >= high:
            raise ValueError("Stratified panel range must have positive width")
        edges = [low + (high - low) * cell_index / size,
                 low + (high - low) * (cell_index + 1) / size]
        actual = bounds[axis]
        if not isinstance(actual, list) or len(actual) != 2:
            raise ValueError("Invalid panel grid cell bounds")
        for left, right in zip(actual, edges):
            _same(left, right, f"grid cell bound {axis}")
        expected_bounds[axis] = edges
        if not edges[0] - 1e-12 <= offsets[axis] <= edges[1] + 1e-12:
            raise ValueError("Panel offset escaped its assigned grid cell")
        if corner:
            _same(offsets[axis], low if cell_index == 0 else high, f"required panel corner {axis}")
    attempts = _integer(proof.get("attempts"), "attempts", 1)
    maximum = _integer(randomization.get("max_attempts", 8), "max_attempts", 1)
    rejected = proof.get("rejected_candidates")
    if not isinstance(rejected, list) or len(rejected) != attempts - 1 or attempts > maximum:
        raise ValueError("Invalid panel grid rejection history")
    if corner and (attempts != 1 or rejected):
        raise ValueError("Required panel corner cannot be replaced by a retry")
    for candidate in rejected:
        if not isinstance(candidate, dict):
            raise ValueError("Invalid rejected panel grid candidate")
        for axis, (low, high) in expected_bounds.items():
            value = _number(candidate.get(f"panel_offset_{axis}_m"), f"rejected candidate {axis}")
            if not low - 1e-12 <= value <= high + 1e-12:
                raise ValueError("Rejected panel candidate escaped its assigned grid cell")


def validate_episode_panel_metadata(collection, metadata):
    """Return verified ``(x, y)`` offsets; require physical evidence for v11."""
    schema = collection.get("raw_schema_version", metadata.get("raw_schema_version", 10))
    new = schema >= 11
    cfg = collection.get("config", {})
    if not new and not any(key in metadata for key in PANEL_EPISODE_CONTEXT):
        # Do not retroactively inject new defaults into old per-episode records.
        if any(_number(cfg.get(f"panel_offset_{axis}_m", 0.), f"config.panel_offset_{axis}_m") != 0. for axis in "xy"):
            raise ValueError("Legacy episode has a nonzero configured panel offset without recorded context")
        return 0., 0.
    offsets = {}
    for axis in "xy":
        key = f"panel_offset_{axis}_m"
        if key not in metadata:
            raise ValueError(f"Missing episode {key}")
        offsets[axis] = _number(metadata[key], key)
    randomization = cfg.get("panel_randomization", {})
    if not isinstance(randomization, dict) or type(randomization.get("enabled", False)) is not bool:
        raise ValueError("Invalid panel_randomization configuration")
    enabled = randomization.get("enabled", False)
    for axis, offset in offsets.items():
        if enabled:
            low = _number(randomization.get(f"min_offset_{axis}_m"), f"min_offset_{axis}_m")
            high = _number(randomization.get(f"max_offset_{axis}_m"), f"max_offset_{axis}_m")
            if low > high or not low - 1e-12 <= offset <= high + 1e-12:
                raise ValueError(f"Episode panel_offset_{axis}_m is outside the configured range")
        else:
            _same(offset, cfg.get(f"panel_offset_{axis}_m", 0.), f"fixed configuration {axis}")
    variation = metadata.get("variation")
    if not isinstance(variation, dict):
        raise ValueError("Missing episode panel variation")
    for axis, offset in offsets.items():
        _same(offset, variation.get(f"panel_offset_{axis}_m"), f"variation {axis}")
    proof = variation.get("panel_randomization")
    if not isinstance(proof, dict) or proof.get("enabled") is not enabled:
        raise ValueError("Missing or inconsistent panel_randomization variation evidence")
    sampling_mode = randomization.get("sampling_mode", "uniform")
    if sampling_mode not in ("uniform", "stratified_grid"):
        raise ValueError("Invalid configured panel sampling mode")
    if enabled and sampling_mode == "stratified_grid":
        _validate_stratified_context(randomization, metadata, proof, offsets)
    elif proof.get("sampling_mode", "uniform") != "uniform":
        raise ValueError("Panel sampling mode differs from configured uniform/disabled sampling")
    layout = metadata.get("panel_layout")
    if not isinstance(layout, dict) or not layout:
        raise ValueError("Missing panel_layout evidence")
    for axis, offset in offsets.items():
        _same(offset, layout.get(f"panel_offset_{axis}_m"), f"panel_layout {axis}")
        if f"panel_offset_{axis}_m" in proof:
            _same(offset, proof[f"panel_offset_{axis}_m"], f"panel_randomization {axis}")
        if enabled:
            for bound in ("min", "max"):
                key = f"{bound}_offset_{axis}_m"
                if key in proof:
                    _same(proof[key], randomization[key], f"panel_randomization {key}")
    _finite_tree(layout, "panel_layout")
    _finite_tree(proof, "variation.panel_randomization")
    if new:
        tolerance = .0002
        positions = layout.get("button_rest_positions_world_m")
        if not isinstance(positions, dict) or set(positions) != {str(floor) for floor in range(24, 36)}:
            raise ValueError("Panel layout must record all twelve measured button rest positions")
        env_offset = metadata.get("env_offset_m")
        if not isinstance(env_offset, (list, tuple)) or len(env_offset) != 3:
            raise ValueError("Missing environment offset for panel layout")
        env_offset = [_number(value, "env_offset_m") for value in env_offset]
        measured_error = 0.
        for floor in range(24, 36):
            actual = positions[str(floor)]
            if not isinstance(actual, (list, tuple)) or len(actual) != 3:
                raise ValueError("Invalid measured button rest position")
            expected = [cfg["button_face_x"] + .003 + offsets["x"] + env_offset[0],
                        cfg["button_column_y"] * (1 if floor < 30 else -1) + offsets["y"] + env_offset[1],
                        cfg["button_bottom_z"] + ((floor - 24) % 6) * cfg["button_pitch_z"] + env_offset[2]]
            measured_error = max(measured_error, math.sqrt(sum((_number(x, "button rest position") - y) ** 2
                                                              for x, y in zip(actual, expected))))
        if measured_error > tolerance:
            raise ValueError("Measured button rest positions disagree with episode panel offset")
        for name in ("max_button_rest_error_m", "max_spring_anchor_error_m"):
            error = _number(layout.get(name), name)
            if not 0 <= error <= tolerance:
                raise ValueError(f"Panel layout error exceeds tolerance: {name}")
        if abs(measured_error - layout["max_button_rest_error_m"]) > 1e-7:
            raise ValueError("Reported panel rest error differs from measured positions")
        bounds = layout.get("panel_world_bounds_m")
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in bounds)):
            raise ValueError("Missing measured panel bounds")
        if any(_number(a, "panel bounds") >= _number(b, "panel bounds") for a, b in zip(*bounds)):
            raise ValueError("Invalid measured panel bounds")
        if abs((bounds[0][1] + bounds[1][1]) / 2 - (offsets["y"] + env_offset[1])) > tolerance:
            raise ValueError("Measured panel bounds disagree with episode offset")
        if layout.get("fixed_rig_unchanged") is not True:
            raise ValueError("Panel randomization moved the fixed camera rig")
        if not isinstance(layout.get("global_camera"), dict) or not layout["global_camera"]:
            raise ValueError("Missing actual global camera projection evidence")
        camera = layout["global_camera"]
        if camera.get("resolution") != [640, 480] or camera.get("all_inside") is not True:
            raise ValueError("Panel bounds are not fully inside the global camera")
        margin = _number(camera.get("margin_px"), "global camera margin")
        required_margin = _number(randomization.get("camera_margin_px", 5.), "configured camera margin")
        if margin < required_margin or margin < 0:
            raise ValueError("Global camera panel margin was reduced")
        points, depths = camera.get("corners_px"), camera.get("corner_depths_m")
        if not isinstance(points, list) or len(points) != 8 or not isinstance(depths, list) or len(depths) != 8:
            raise ValueError("Missing eight measured panel corner projections")
        for point, depth in zip(points, depths):
            if not isinstance(point, (list, tuple)) or len(point) != 2 or _number(depth, "corner depth") <= 0:
                raise ValueError("Invalid panel corner projection")
            u, v = (_number(value, "corner pixel") for value in point)
            if not (margin <= u <= 639 - margin and margin <= v <= 479 - margin):
                raise ValueError("Projected panel corner exceeds global camera margin")
    return offsets["x"], offsets["y"]
