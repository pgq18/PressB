"""Elevator-scene placement and defaults for the fixed D435 camera."""

from __future__ import annotations

import numpy as np

from pressb.robots.piper.fixed_camera import build_fixed_camera


def build_global_camera(stage, cfg=None):
    """Build the elevator scene's static camera rig and recorder metadata.

    Config: ``global_camera_eye``, ``global_camera_target``,
    ``global_camera_resolution``, ``global_camera_support_surface``,
    ``global_camera_stand_footprint_m`` and ``global_camera_head_riser_m``.
    Position vectors use world metres. The camera must remain on the +Y (left)
    side of the elevator scene.
    """
    cfg = cfg or {}
    eye = np.asarray(cfg.get("global_camera_eye", [-.28, .24, 1.05]), dtype=float)
    target = np.asarray(cfg.get("global_camera_target", [.43, 0., 1.0675]), dtype=float)
    support_surface = cfg.get("global_camera_support_surface", "tabletop")
    if support_surface not in ("tabletop", "floor"):
        raise ValueError("global_camera_support_surface must be tabletop or floor")
    support_z = float(cfg.get("table_height", .76)) if support_surface == "tabletop" else 0.
    footprint = float(cfg.get("global_camera_stand_footprint_m", .18))
    head_riser = float(cfg.get("global_camera_head_riser_m", .05))
    if not np.isfinite([support_z, footprint, head_riser]).all() or footprint <= 0 or head_riser < .02:
        raise ValueError("Camera stand dimensions must be finite, footprint positive and head riser at least 20 mm")
    if any(p.shape != (3,) or not np.isfinite(p).all() for p in (eye, target)):
        raise ValueError("global_camera_eye and global_camera_target must be finite XYZ vectors")
    if eye[1] <= 0:
        raise ValueError("The global D435 must be on the left side of the scene (+Y)")
    table_edge = float(cfg.get("table_edge_x", -.14))
    table_depth = float(cfg.get("table_depth", .75))
    table_width = float(cfg.get("table_width", 1.))
    support_bounds = [[table_edge - table_depth, -table_width / 2, support_z],
                      [table_edge, table_width / 2, support_z]] if support_surface == "tabletop" else None
    return build_fixed_camera(
        stage, eye=eye, target=target,
        resolution=cfg.get("global_camera_resolution", [640, 480]),
        support_surface=support_surface, support_z=support_z,
        footprint=footprint, head_riser=head_riser, support_bounds=support_bounds,
    )
