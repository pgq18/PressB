#!/usr/bin/env python3
"""Change the simulated base mounting height while keeping the room fixed.

CPU-only coordinate re-expression: no simulator, IK, robot driver, or hardware.
The wrist remains link-local; the table and tabletop object coordinates remain
unchanged. Replay an existing plan to isolate mounting-height effects; making a
new tabletop-relative plan would compensate for the new height through IK.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path


def _height(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("robot_base_height_above_table_m must be finite and nonnegative")
    return value


def adjust_mount_height(cfg, height_m):
    """Return a deep copy with a new mount height and fixed world fixtures.

    Both camera matrix translations and their redundant eye/target descriptions
    are updated, so the rendered optical centre and visual tripod stay together.
    Existing provenance is retained; the appended record describes a coordinate
    conversion, not an accepted measurement or a recalibration.
    """
    if cfg.get("task") != "playground":
        raise ValueError("This conversion supports task=playground only")
    previous = _height(cfg.get("robot_base_height_above_table_m", 0.))
    height = _height(height_m)
    result = copy.deepcopy(cfg)
    # Return exactly the same metadata/configuration for the control condition.
    if height == previous:
        return result
    offset = previous - height
    updated = []

    def shift_vector(container, key, path):
        if key not in container:
            return
        value = container[key]
        if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(
                isinstance(v, (float, int)) and math.isfinite(v) for v in value):
            raise ValueError(f"{path} must contain three finite coordinates")
        container[key] = list(value)
        container[key][2] += offset
        updated.append(path)

    for name in ("up_camera", "overview_camera"):
        camera = result.get(name)
        if camera is None:
            continue
        for key in ("eye_base_m", "target_base_m"):
            shift_vector(camera, key, f"{name}.{key}")
        if "base_from_camera_usd" in camera:
            matrix = camera["base_from_camera_usd"]
            if (not isinstance(matrix, (list, tuple)) or len(matrix) != 4 or
                    any(not isinstance(row, (list, tuple)) or len(row) != 4 or
                        any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in row)
                        for row in matrix) or list(matrix[3]) != [0, 0, 0, 1]):
                raise ValueError(f"{name}.base_from_camera_usd must be a finite homogeneous 4x4 matrix")
            camera["base_from_camera_usd"] = [list(row) for row in matrix]
            camera["base_from_camera_usd"][2][3] += offset
            updated.append(f"{name}.base_from_camera_usd[2][3]")
    for i, prop in enumerate(result.get("static_props", [])):
        shift_vector(prop, "center_base_m", f"static_props[{i}].center_base_m")
    for i, light in enumerate(result.get("lighting", {}).get("area_lights", [])):
        for key in ("position_base_m", "target_base_m"):
            shift_vector(light, key, f"lighting.area_lights[{i}].{key}")
    result["robot_base_height_above_table_m"] = height
    result.setdefault("reconstruction", {})["mount_coordinate_reexpression"] = {
        "previous_height_m": previous,
        "height_m": height,
        "fixed_world_local_z_offset_m": offset,
        "updated_coordinate_fields": updated,
        "status": "Coordinate conversion only; not a measured or accepted mounting height.",
        "invariants": "Table, dynamic object XY/Z, fixed camera world poses, room props and lights unchanged; wrist link-local transform and intrinsics unchanged.",
        "replay": "Use the same recorded joint plan. New tabletop-height IK plans compensate for the mount change.",
        "hardware_actions": False,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--height-m", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = adjust_mount_height(json.loads(args.config.read_text()), args.height_m)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental overwriting of a source/default.
    with args.output.open("x") as stream:
        json.dump(cfg, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "height_m": args.height_m,
                      "hardware_actions": False}))


if __name__ == "__main__":
    main()
