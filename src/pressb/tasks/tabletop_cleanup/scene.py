"""Compose the unchanged playground workbench with a static bin and compound props."""
from __future__ import annotations
from copy import deepcopy
import math
import numpy as np


def build_scene(stage, config):
    from pressb.tasks.playground.scene import build_scene as build_workspace
    from .assets import create_bin, create_object
    cfg = deepcopy(config)
    cfg["objects"] = []
    scene = build_workspace(stage, cfg)
    base = np.asarray(scene["robot_base_position_world"], dtype=float)
    root, table = scene["root_path"], float(scene["table_top_z"])
    box = cfg["cleanup_bin"]
    half = math.radians(box["yaw_deg"])/2
    position = [*(base[:2]+box["center_xy"]), table+box["size_xyz_m"][2]/2]
    scene["bin"] = create_bin(stage, root+"/Bin", position=position,
        orientation_wxyz=[math.cos(half), 0., 0., math.sin(half)],
        size_xyz=box["size_xyz_m"], wall_thickness=box["wall_thickness_m"])
    scene["bin"]["yaw_deg"] = box["yaw_deg"]
    scene["collision_paths"].extend(scene["bin"]["collision_paths"])
    from .geometry import KINDS, asset_spec
    for kind in KINDS:
        obj = cfg["cleanup_objects"][kind]
        spec = asset_spec(kind, scale=obj["scale"])
        bottom = float(np.asarray(spec["support_vertices"])[:, 2].min())
        position = [*(base[:2]+obj["center_xy"]), table-bottom+.0005]
        half = math.radians(obj["yaw_deg"])/2
        scene["objects"][kind] = create_object(stage, root+"/Objects/"+kind, kind,
            position=position, orientation_wxyz=[math.cos(half), 0., 0., math.sin(half)],
            mass_kg=obj["mass_kg"], scale=obj["scale"])
        scene["collision_paths"].extend(scene["objects"][kind]["collision_paths"])
    scene["task_id"] = "tabletop_cleanup"
    scene["outcome_semantics"] = "Exactly two active objects: full geometry inside bin with grasp history and physical support"
    return scene
