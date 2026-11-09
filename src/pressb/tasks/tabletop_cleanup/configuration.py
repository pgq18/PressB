"""Exact-two object selection, footprint-safe layout sampling and task identity."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import math
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from pressb.runtime.contract import finite_array, fingerprint
from pressb.tasks.playground.backend import ROOT, SOURCE_FILES as WORKSPACE_SOURCES, load_config as workspace_config
from .geometry import KINDS, asset_spec
from .task import INSTRUCTION, number, outcome_config

DEFAULT_OBJECTS = {
    "gamepad": dict(scale=1., center_xy=[.350157101832, .038150647637], yaw_deg=1.326457748),
    "speaker": dict(scale=1., center_xy=[.3600054035, -.2129193382], yaw_deg=1.4599494),
    "lidar": dict(scale=1., center_xy=[.3552321047, -.0871805975], yaw_deg=.5298066)}
# Fixed tabletop box pose remeasured with the refined global camera on 2026-11-09.
DEFAULT_BIN = dict(center_xy=[.3396856, .21861745], yaw_deg=2.070158,
                   size_xyz_m=[.222, .148, .049], wall_thickness_m=.004)
DEFAULT_RANDOMIZATION = dict(enabled=True, pool=list(KINDS), fixed_subset=None,
    x_range_m=[.24, .48], y_range_m=[-.25, .19], yaw_range_deg=[-180., 180.], clearance_m=.01, max_attempts=2048)
SOURCE_FILES = (*WORKSPACE_SOURCES, "tasks/tabletop_cleanup/__init__.py", "tasks/tabletop_cleanup/configuration.py",
    "tasks/tabletop_cleanup/task.py", "tasks/tabletop_cleanup/parallel.py", "tasks/tabletop_cleanup/scene.py",
    "tasks/tabletop_cleanup/assets.py", "tasks/tabletop_cleanup/geometry.py")


def finite_yaw(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("yaw_deg must be finite")
    return float(value)


def subset(value, pool):
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or any(not isinstance(k, str) for k in value)
            or len(set(value)) != 2 or any(k not in pool for k in value)):
        raise ValueError("Exactly two distinct object kinds from the configured pool are required")
    return [kind for kind in KINDS if kind in value]


def footprint(vertices, xy, yaw):
    vertices = np.asarray(vertices, dtype=float)
    rotated = vertices @ Rotation.from_euler("z", yaw, degrees=True).as_matrix().T
    points = rotated[:, :2]+np.asarray(xy)
    return points[ConvexHull(points).vertices]


def separated(a, b, clearance):
    """Conservative convex-footprint separation using all edge normals."""
    for polygon in (a, b):
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            edge = end-start
            norm = float(np.linalg.norm(edge))
            if norm < 1e-12:
                continue
            axis = np.array([-edge[1], edge[0]])/norm
            pa, pb = a @ axis, b @ axis
            if float(pb.min()-pa.max()) >= clearance or float(pa.min()-pb.max()) >= clearance:
                return True
    return False


def bin_footprint(config):
    box = config["bin"]
    x, y, _ = box["size_xyz_m"]
    vertices = np.array([[-x/2, -y/2, 0], [x/2, -y/2, 0], [x/2, y/2, 0], [-x/2, y/2, 0]])
    return footprint(vertices, box["center_xy"], box["yaw_deg"])


def validate_layout(config, objects):
    subset(list(objects), config["randomization"]["pool"])
    region, scene = config["randomization"], config["scene"]
    table = scene.get("table", {})
    xmin = float(scene.get("table_x_min", table.get("x_min", -.12)))
    xmax = xmin+float(scene.get("table_depth", table.get("depth", 1.05)))
    halfwidth = float(scene.get("table_width", table.get("width", 1.6)))/2
    polygons = {}
    for kind, pose in objects.items():
        xy = finite_array(pose["center_xy"], (2,), kind+".center_xy", dtype=np.float64)
        yaw = finite_yaw(pose["yaw_deg"])
        points = footprint(config["asset_specs"][KINDS.index(kind)]["support_vertices"], xy, yaw)
        for j, bounds in enumerate((region["x_range_m"], region["y_range_m"])):
            if points[:, j].min() < bounds[0]-1e-12 or points[:, j].max() > bounds[1]+1e-12:
                raise ValueError(f"{kind} full footprint exceeds sampling bounds")
        if (points[:, 0].min() < xmin or points[:, 0].max() > xmax or
                points[:, 1].min() < -halfwidth or points[:, 1].max() > halfwidth):
            raise ValueError(f"{kind} is not fully supported by tabletop")
        if not separated(points, bin_footprint(config), region["clearance_m"]):
            raise ValueError(f"{kind} initial footprint overlaps the fixed bin or clearance")
        polygons[kind] = points
    first, second = polygons.values()
    if not separated(first, second, region["clearance_m"]):
        raise ValueError("Object initial footprints overlap or violate clearance")


def load_config(config):
    if not isinstance(config, dict):
        raise ValueError("Cleanup config must be a mapping")
    raw = deepcopy(config)
    options = raw.pop("objects", {})
    if not isinstance(options, dict) or set(options)-set(KINDS):
        raise ValueError("objects must map gamepad/speaker/lidar to asset settings")
    objects, assets = {}, []
    for kind in KINDS:
        updates = options.get(kind, {})
        if not isinstance(updates, dict) or set(updates)-{"scale", "mass_kg", "center_xy", "yaw_deg"}:
            raise ValueError("Object config supports scale, mass_kg, center_xy, yaw_deg")
        obj = {**deepcopy(DEFAULT_OBJECTS[kind]), **updates}
        obj["scale"] = number(obj["scale"], kind+".scale")
        spec = asset_spec(kind, scale=obj["scale"])
        obj["mass_kg"] = number(obj.get("mass_kg", spec["mass_kg"]), kind+".mass_kg")
        obj["center_xy"] = finite_array(obj["center_xy"], (2,), kind+".center_xy", dtype=np.float64).tolist()
        obj["yaw_deg"] = finite_yaw(obj["yaw_deg"])
        objects[kind] = obj
        assets.append(spec)
    values = raw.pop("bin", {})
    if not isinstance(values, dict) or set(values)-set(DEFAULT_BIN):
        raise ValueError("Unsupported bin configuration")
    box = {**deepcopy(DEFAULT_BIN), **values}
    size = finite_array(box["size_xyz_m"], (3,), "bin size", dtype=np.float64)
    if np.any(size <= 0):
        raise ValueError("Bin dimensions must be positive")
    box["size_xyz_m"] = size.tolist()
    box["center_xy"] = finite_array(box["center_xy"], (2,), "bin center", dtype=np.float64).tolist()
    box["yaw_deg"] = finite_yaw(box["yaw_deg"])
    box["wall_thickness_m"] = number(box["wall_thickness_m"], "bin wall thickness")
    if np.any(size <= 2*box["wall_thickness_m"]):
        raise ValueError("Bin wall thickness leaves no cavity")
    values = raw.pop("randomization", {})
    if not isinstance(values, dict) or set(values)-set(DEFAULT_RANDOMIZATION):
        raise ValueError("Unknown randomization fields; every episode has exactly two objects")
    random = {**deepcopy(DEFAULT_RANDOMIZATION), **values}
    if type(random["enabled"]) is not bool:
        raise ValueError("randomization.enabled must be boolean")
    pool = random["pool"]
    if (not isinstance(pool, list) or not 2 <= len(pool) <= 3 or any(not isinstance(k, str) or k not in KINDS for k in pool)
            or len(set(pool)) != len(pool)):
        raise ValueError("pool must contain two or three distinct known object kinds")
    random["pool"] = [k for k in KINDS if k in pool]
    if random["fixed_subset"] is not None:
        random["fixed_subset"] = subset(random["fixed_subset"], pool)
    for key in ("x_range_m", "y_range_m", "yaw_range_deg"):
        bounds = finite_array(random[key], (2,), key, dtype=np.float64)
        if bounds[1] < bounds[0] or (key != "yaw_range_deg" and bounds[1] == bounds[0]):
            raise ValueError("Randomization bounds must be ordered")
        random[key] = bounds.tolist()
    random["clearance_m"] = number(random["clearance_m"], "clearance_m", zero=True)
    if type(random["max_attempts"]) is not int or random["max_attempts"] < 1:
        raise ValueError("max_attempts must be a positive integer")
    outcome = outcome_config(raw.pop("outcome", {}))
    reward = raw.pop("reward", {})
    if not isinstance(reward, dict) or set(reward)-{"success", "failure"}:
        raise ValueError("reward supports success and failure")
    reward = {**dict(success=1., failure=0.), **reward}
    for key, v in reward.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError("Rewards must be finite")
        reward[key] = float(v)
    terminate = raw.pop("terminate_on_success", True)
    if type(terminate) is not bool:
        raise ValueError("terminate_on_success must be boolean")
    raw.setdefault("scene_config", "configs/calibration/playground_20261109_global_refined.json")
    overrides = deepcopy(raw.get("scene_overrides", {}))
    if not isinstance(overrides, dict) or "objects" in overrides:
        raise ValueError("Configure cleanup objects through top-level objects")
    overrides.update(root_path="/World/TabletopCleanup", task="tabletop_cleanup", task_instruction=INSTRUCTION, objects=[])
    raw["scene_overrides"] = overrides
    raw.setdefault("log_dir", str(ROOT/"outputs/tabletop_cleanup_parallel"))
    raw.setdefault("max_episode_seconds", 120.)
    result = workspace_config(raw)
    result.update(objects=objects, asset_specs=assets, bin=box, randomization=random,
                  outcome=outcome, reward=reward, terminate_on_success=terminate)
    result["scene"].update(cleanup_objects=deepcopy(objects), cleanup_bin=deepcopy(box))
    if outcome["gripper_release_width_m"] > result["scene"]["gripper_max_width_m"]:
        raise ValueError("Configured release width exceeds gripper travel")
    # A fixed box must lie completely on the existing table, irrespective of
    # the object spawn rectangle (which intentionally excludes its footprint).
    points = bin_footprint(result)
    scene = result["scene"]
    xmin, xmax = scene["table_x_min"], scene["table_x_min"]+scene["table_depth"]
    if (points[:, 0].min() < xmin or points[:, 0].max() > xmax or
            np.max(np.abs(points[:, 1])) > scene["table_width"]/2):
        raise ValueError("Fixed bin must be fully supported by the tabletop")
    if not random["enabled"]:
        candidates = [random["fixed_subset"]] if random["fixed_subset"] else [list(pair) for pair in
            combinations(random["pool"], 2)]
        for selected in candidates:
            validate_layout(result, {k: {key: objects[k][key] for key in ("center_xy", "yaw_deg")} for k in selected})
    return result


def sample_objects(config, env_id, seed, overrides=None):
    if type(env_id) is not int or env_id < 0 or type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Reset requires nonnegative integer env_id and seed in [0,2**63)")
    random, rng = config["randomization"], np.random.default_rng(np.random.SeedSequence([seed, env_id]))
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError("objects reset must be a mapping for exactly two objects")
        selected = subset(list(overrides), random["pool"])
        overrides = deepcopy(overrides)
    else:
        selected = random["fixed_subset"] or subset(rng.choice(random["pool"], size=2, replace=False).tolist(), random["pool"])
        overrides = {}
    for kind, value in overrides.items():
        if not isinstance(value, dict) or set(value)-{"center_xy", "yaw_deg"}:
            raise ValueError("Object resets support center_xy and yaw_deg only")
        if "center_xy" in value:
            value["center_xy"] = finite_array(value["center_xy"], (2,), kind+".center_xy", dtype=np.float64).tolist()
        if "yaw_deg" in value:
            value["yaw_deg"] = finite_yaw(value["yaw_deg"])
    last_error = None
    for _ in range(random["max_attempts"]):
        poses = {}
        rejected_yaw = False
        for kind in selected:
            obj, override = config["objects"][kind], overrides.get(kind, {})
            yaw = override.get("yaw_deg", obj["yaw_deg"] if "center_xy" in override or not random["enabled"]
                               else float(rng.uniform(*random["yaw_range_deg"])))
            local = footprint(config["asset_specs"][KINDS.index(kind)]["support_vertices"], [0., 0.], yaw)
            bounds = [(random[key][0]-local[:, j].min(), random[key][1]-local[:, j].max())
                      for j, key in enumerate(("x_range_m", "y_range_m"))]
            if any(high < low for low, high in bounds):
                last_error = ValueError("Sampling domain cannot contain this rotated object")
                random_yaw = (random["enabled"] and "yaw_deg" not in override and "center_xy" not in override
                              and random["yaw_range_deg"][0] < random["yaw_range_deg"][1])
                if random_yaw:
                    rejected_yaw = True
                    break
                raise last_error
            xy = override.get("center_xy", [float(rng.uniform(low, high)) for low, high in bounds]
                              if random["enabled"] else obj["center_xy"])
            poses[kind] = {"center_xy": list(xy), "yaw_deg": float(yaw)}
        if rejected_yaw:
            continue
        try:
            validate_layout(config, poses)
            return {"active_names": list(selected), "active_mask": [k in selected for k in KINDS], "objects": poses}
        except ValueError as error:
            last_error = error
            if not random["enabled"] or all("center_xy" in overrides.get(k, {}) for k in selected):
                raise
    raise ValueError(f"No nonoverlapping layout after {random['max_attempts']} proposals: {last_error}")


def task_identity(config):
    scene = deepcopy(config["scene"])
    assets = {key: hashlib.sha256(Path(scene.pop(key)).read_bytes()).hexdigest() for key in ("robot_usd", "robot_urdf")}
    sources = {}
    for name in SOURCE_FILES:
        content = (ROOT/"src/pressb"/name).read_bytes()
        sources[name] = dict(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    semantics = {key: deepcopy(config[key]) for key in ("horizon", "control_hz", "physics_dt", "max_episode_seconds", "gamma",
        "settling_ticks", "smoothing", "model_inputs", "objects", "asset_specs", "bin", "randomization", "outcome", "reward", "terminate_on_success")}
    semantics.update(scene=scene, assets=assets, instruction=INSTRUCTION, active_count=2, outcome_version="full_geometry_support_graph_cleanup_v1")
    return dict(task_config_sha256=fingerprint(semantics), source_identity=sources,
        source_revision="sha256:"+fingerprint(sources), asset_sha256=assets,
        asset_scope="Root robot USD, URDF and authored compound geometry; external USD references not recursively hashed")
