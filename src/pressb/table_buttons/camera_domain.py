"""Full tabletop button-centre domain of the inherited pinhole camera.

This computes visibility and tabletop support only, before robot reachability,
path collision, or occlusion checks. It has no simulator dependency.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _clip_polygon(polygon, line):
    """Clip a counterclockwise XY polygon against A*x + B*y + C >= 0."""
    if not len(polygon):
        return polygon
    result = []
    values = polygon @ line[:2] + line[2]
    for index, point in enumerate(polygon):
        next_index = (index + 1) % len(polygon)
        following = polygon[next_index]
        here, there = values[index], values[next_index]
        if here >= 0:
            result.append(point)
        if (here >= 0) != (there >= 0):
            result.append(point + here / (here - there) * (following - point))
    if not result:
        return np.empty((0, 2), dtype=float)
    # A corner already on a clip boundary can otherwise appear twice.
    polygon = np.asarray(result, dtype=float)
    distinct = np.linalg.norm(polygon - np.roll(polygon, 1, axis=0), axis=1) > 1e-12
    return polygon[distinct]


def camera_center_domain(cfg: Mapping) -> dict:
    """Return the complete visible centre polygon in robot-base XY metres.

    The housing is enclosed by the same axis-aligned prism used by
    :func:`button_in_camera`: radius in X/Y, from tabletop z=0 to cap top.
    Each image-frustum halfspace is reduced analytically over that prism,
    then intersected with the tabletop inset by the housing radius. Configured
    sampling bounds are intentionally ignored.

    ``bounds_xy_m`` is ``[[xmin, xmax], [ymin, ymax]]`` or ``None`` for an
    empty domain. Vertices are counterclockwise. ``halfspaces`` use the
    convention ``A*x + B*y + C >= 0``. A 1 nm positive-depth guard avoids
    undefined projection at the camera plane. Candidate positions still need
    robot/path checks and the normal visibility predicate at runtime.
    """
    camera = cfg["up_camera"]
    geometry = cfg.get("button_geometry", {})
    radius = float(geometry.get("base_radius_m", .022))
    base_height = float(geometry.get("base_height_m", .017))
    cap_height = float(geometry.get("cap_height_m", .008))
    top = base_height + cap_height
    margin = float(cfg.get("position_sampling", {}).get("pixel_margin", 16.))
    table_x_min = float(cfg["table_x_min"])
    table_depth = float(cfg["table_depth"])
    table_width = float(cfg["table_width"])
    values = [radius, base_height, cap_height, margin, table_x_min, table_depth, table_width]
    if (not np.isfinite(values).all() or min(radius, base_height, cap_height, table_depth, table_width) <= 0
            or margin < 0):
        raise ValueError("Expected positive finite button/table dimensions and a nonnegative pixel margin")

    matrix = np.asarray(camera["base_from_camera_usd"], dtype=float)
    resolution = np.asarray(camera["resolution"], dtype=float)
    fx, fy, cx, cy = [float(camera[key]) for key in ("fx", "fy", "cx", "cy")]
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-12, rtol=0.)):
        raise ValueError("Expected a finite affine 4x4 camera transform")
    if (resolution.shape != (2,) or not np.isfinite(resolution).all() or np.any(resolution <= 1)
            or not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0):
        raise ValueError("Expected positive finite focal lengths and a valid image resolution")
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("Camera transform must be invertible") from error

    # USD camera +Y is up, -Z is forward; image v increases downwards.
    x_row, y_row, depth_row = inverse[0], inverse[1], -inverse[2]
    width, height = resolution
    min_depth = 1e-9
    frustum_planes = [
        ("camera_left", fx*x_row + (cx-margin)*depth_row),
        ("camera_right", (width-1-margin-cx)*depth_row - fx*x_row),
        ("camera_top", (cy-margin)*depth_row - fy*y_row),
        ("camera_bottom", (height-1-margin-cy)*depth_row + fy*y_row),
        ("camera_positive_depth", depth_row - [0., 0., 0., min_depth]),
    ]
    center_limits = np.array([[table_x_min+radius, table_x_min+table_depth-radius],
                              [-table_width/2+radius, table_width/2-radius]])
    table_planes = [
        ("table_x_min", np.array([1., 0., -center_limits[0, 0]])),
        ("table_x_max", np.array([-1., 0., center_limits[0, 1]])),
        ("table_y_min", np.array([0., 1., -center_limits[1, 0]])),
        ("table_y_max", np.array([0., -1., center_limits[1, 1]])),
    ]
    camera_planes = []
    for name, row in frustum_planes:
        constant = row[3] - radius*(abs(row[0])+abs(row[1])) + min(0., row[2]*top)
        camera_planes.append((name, np.r_[row[:2], constant]))

    polygon = np.empty((0, 2), dtype=float)
    if np.all(center_limits[:, 0] < center_limits[:, 1]) and np.all(resolution-1 > 2*margin):
        xmin, xmax = center_limits[0]
        ymin, ymax = center_limits[1]
        polygon = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        for _, line in camera_planes:
            polygon = _clip_polygon(polygon, line)
    area = 0.
    if len(polygon) >= 3:
        area = float(abs(np.sum(polygon[:, 0]*np.roll(polygon[:, 1], -1)
                                    - polygon[:, 1]*np.roll(polygon[:, 0], -1)))/2)
    if area <= 1e-15:
        polygon = np.empty((0, 2), dtype=float)
        area = 0.

    return {
        "frame": "robot base XY, tabletop z=0",
        "polygon_xy_m": polygon.tolist(),
        "bounds_xy_m": (np.column_stack((polygon.min(axis=0), polygon.max(axis=0))).tolist()
                        if len(polygon) else None),
        "area_m2": area,
        "table_center_limits_xy_m": center_limits.tolist(),
        "halfspace_convention": "A*x + B*y + C >= 0",
        "halfspaces": [{"boundary": name, "coefficients_A_B_C": line.tolist()}
                       for name, line in table_planes + camera_planes],
        "geometry": {"enclosure": "axis_aligned_housing_prism", "radius_m": radius,
                     "base_height_m": base_height, "cap_height_m": cap_height,
                     "bottom_z_m": 0., "top_z_m": top},
        "camera": {"resolution": resolution.tolist(), "pixel_margin": margin,
                   "min_depth_m": min_depth},
        "checks_included": ["full_housing_image_bounds", "full_housing_tabletop_support"],
        "checks_excluded": ["robot_reachability", "path_collision", "occlusion"],
    }
