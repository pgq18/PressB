"""CPU-only configuration, bounded sampling and provenance for cube stacking."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import numpy as np

from pressb.runtime.contract import finite_array, fingerprint
from pressb.robots.piper.workspace_pose import apply_workspace_home
from pressb.tasks.playground.backend import ROOT, SOURCE_FILES as WORKSPACE_SOURCES, load_config as workspace_config
from .task import COLORS, INSTRUCTION, _number, footprint, intersection_area, outcome_config

DEFAULT_CUBES = {
    "red": dict(edge_m=.029, center_xy=[.331263, .102522], yaw_deg=-10.51, mass_kg=.025,
                color=[.57, .018, .025], roughness=.22),
    "blue": dict(edge_m=.030, center_xy=[.326851, -.083409], yaw_deg=30.36, mass_kg=.025,
                 color=[.006, .35, .55], roughness=.22)}
DEFAULT_RANDOMIZATION = dict(enabled=True, x_range_m=[.25, .45], y_range_m=[-.16, .16],
    yaw_range_deg=[0., 0.], min_center_distance_m=.08, max_attempts=1024)
SOURCE_FILES = (*WORKSPACE_SOURCES, "tasks/cube_stack/__init__.py", "tasks/cube_stack/configuration.py",
                "tasks/cube_stack/task.py", "tasks/cube_stack/parallel.py", "robots/piper/workspace_pose.py")


def _range(value, name, *, equal=False):
    result = finite_array(value, (2,), name, dtype=np.float64)
    if result[0] > result[1] or (not equal and result[0] == result[1]):
        raise ValueError(f"{name} bounds must be increasing")
    return result.tolist()


def validate_layout(config, objects):
    """Require full rotated footprints inside the sampling rectangle/table."""
    if set(objects) != set(COLORS):
        raise ValueError("A layout requires red and blue objects")
    region, scene = config["randomization"], config["scene"]
    table = scene.get("table", {})
    xmin = float(scene.get("table_x_min", table.get("x_min", -.12)))
    xmax = xmin+float(scene.get("table_depth", table.get("depth", 1.05)))
    halfwidth = float(scene.get("table_width", table.get("width", 1.6)))/2
    polygons = {}
    for color in COLORS:
        obj, cube = objects[color], config["cubes"][color]
        xy = finite_array(obj["center_xy"], (2,), color+".center_xy", dtype=np.float64)
        yaw = obj["yaw_deg"]
        if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
            raise ValueError("Object yaw_deg must be finite")
        vertices = footprint(xy, cube["edge_m"], yaw)
        for dimension, bounds in enumerate((region["x_range_m"], region["y_range_m"])):
            if vertices[:, dimension].min() < bounds[0]-1e-12 or vertices[:, dimension].max() > bounds[1]+1e-12:
                raise ValueError(f"{color} cube footprint exceeds configured sampling bounds")
        if (vertices[:, 0].min() < xmin or vertices[:, 0].max() > xmax or
                vertices[:, 1].min() < -halfwidth or vertices[:, 1].max() > halfwidth):
            raise ValueError(f"{color} cube must be fully supported by the tabletop")
        polygons[color] = vertices
    separation = float(np.linalg.norm(np.subtract(objects["red"]["center_xy"], objects["blue"]["center_xy"])))
    if separation+1e-12 < region["min_center_distance_m"]:
        raise ValueError("Cube center separation is below min_center_distance_m")
    if intersection_area(polygons["red"], polygons["blue"]) > 1e-12:
        raise ValueError("Cube initial footprints must not overlap")


def load_config(config):
    if not isinstance(config, dict):
        raise ValueError("cube_stack configuration must be a mapping")
    raw = deepcopy(config)
    initial_pose = raw.pop("initial_pose", None)
    cube_options = raw.pop("cubes", {})
    if not isinstance(cube_options, dict) or set(cube_options)-set(COLORS):
        raise ValueError("cubes supports red and blue only")
    cubes = {}
    for color in COLORS:
        overrides = cube_options.get(color, {})
        if not isinstance(overrides, dict) or set(overrides)-set(DEFAULT_CUBES[color]):
            raise ValueError(f"Unsupported cubes.{color} fields")
        cube = {**deepcopy(DEFAULT_CUBES[color]), **overrides}
        cube["edge_m"] = _number(cube["edge_m"], color+".edge_m")
        cube["mass_kg"] = _number(cube["mass_kg"], color+".mass_kg")
        cube["center_xy"] = finite_array(cube["center_xy"], (2,), color+".center_xy", dtype=np.float64).tolist()
        if isinstance(cube["yaw_deg"], bool) or not isinstance(cube["yaw_deg"], (int, float)) or not math.isfinite(cube["yaw_deg"]):
            raise ValueError("Cube yaw_deg must be finite")
        cube["yaw_deg"] = float(cube["yaw_deg"])
        rgb = finite_array(cube["color"], (3,), color+".color", dtype=np.float64)
        if np.any(rgb < 0) or np.any(rgb > 1):
            raise ValueError("Cube colors must be in [0,1]")
        cube["color"] = rgb.tolist()
        cube["roughness"] = _number(cube["roughness"], color+".roughness", zero=True)
        if cube["roughness"] > 1:
            raise ValueError("Cube roughness must be in [0,1]")
        cubes[color] = cube
    random = raw.pop("randomization", {})
    if not isinstance(random, dict) or set(random)-set(DEFAULT_RANDOMIZATION):
        raise ValueError(f"randomization supports only {sorted(DEFAULT_RANDOMIZATION)}")
    random = {**deepcopy(DEFAULT_RANDOMIZATION), **random}
    if type(random["enabled"]) is not bool:
        raise ValueError("randomization.enabled must be boolean")
    for key in ("x_range_m", "y_range_m", "yaw_range_deg"):
        random[key] = _range(random[key], key, equal=key == "yaw_range_deg")
    random["min_center_distance_m"] = _number(random["min_center_distance_m"], "min_center_distance_m", zero=True)
    if type(random["max_attempts"]) is not int or random["max_attempts"] < 1:
        raise ValueError("max_attempts must be a positive integer")
    outcome = outcome_config(raw.pop("outcome", {}))
    reward = raw.pop("reward", {})
    if not isinstance(reward, dict) or set(reward)-{"success", "failure"}:
        raise ValueError("reward supports success and failure")
    reward = {**dict(success=1., failure=0.), **reward}
    for key, value in reward.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Rewards must be finite numbers")
        reward[key] = float(value)
    terminate = raw.pop("terminate_on_success", True)
    if type(terminate) is not bool:
        raise ValueError("terminate_on_success must be boolean")
    raw.setdefault("scene_config", "configs/calibration/playground_20261113_appearance.json")
    scene_overrides = deepcopy(raw.get("scene_overrides", {}))
    if not isinstance(scene_overrides, dict):
        raise ValueError("scene_overrides must be a mapping")
    if "objects" in scene_overrides:
        raise ValueError("Configure task objects through cubes, not scene_overrides.objects")
    scene_overrides.update(root_path="/World/CubeStack", task="cube_stack", task_instruction=INSTRUCTION,
        objects=[dict(name=color, size=[cube["edge_m"]]*3, center_xy=cube["center_xy"], yaw_deg=cube["yaw_deg"],
                      mass=cube["mass_kg"], color=cube["color"], roughness=cube["roughness"], clearance=.0005)
                 for color, cube in cubes.items()])
    raw["scene_overrides"] = scene_overrides
    raw.setdefault("log_dir", str(ROOT/"outputs/cube_stack_parallel"))
    result = workspace_config(raw)
    result["initial_pose"] = apply_workspace_home(result["scene"], random, initial_pose,
        explicit_joint_override=any(key in scene_overrides for key in
            ("initial_state_joint_rad_m", "initial_joint_positions_rad", "initial_gripper_width_m")))
    result.update(cubes=cubes, randomization=random, outcome=outcome, reward=reward, terminate_on_success=terminate)
    if cubes["red"]["edge_m"]+outcome["gripper_release_margin_m"] > result["scene"]["gripper_max_width_m"]:
        raise ValueError("Red cube is too wide for the configured gripper release margin")
    validate_layout(result, {color: {key: cube[key] for key in ("center_xy", "yaw_deg")} for color, cube in cubes.items()})
    return result


def sample_objects(config, env_id, seed, overrides=None):
    if type(seed) is not int or not 0 <= seed < 2**63 or type(env_id) is not int or env_id < 0:
        raise ValueError("Reset needs nonnegative integer env_id and seed in [0,2**63)")
    overrides = {} if overrides is None else deepcopy(overrides)
    if not isinstance(overrides, dict) or set(overrides)-set(COLORS):
        raise ValueError("objects supports red and blue only")
    for color, values in overrides.items():
        if not isinstance(values, dict) or set(values)-{"center_xy", "yaw_deg"}:
            raise ValueError("Object overrides support center_xy and yaw_deg only")
        if "center_xy" in values:
            values["center_xy"] = finite_array(values["center_xy"], (2,), color+".center_xy", dtype=np.float64).tolist()
        if "yaw_deg" in values and (isinstance(values["yaw_deg"], bool) or not isinstance(values["yaw_deg"], (int, float))
                                    or not math.isfinite(values["yaw_deg"])):
            raise ValueError("Object yaw_deg must be finite")
    rng = np.random.default_rng(np.random.SeedSequence([seed, env_id]))
    random, cubes = config["randomization"], config["cubes"]
    last_error = None
    for _ in range(random["max_attempts"]):
        result = {}
        for color in COLORS:
            changes, cube = overrides.get(color, {}), cubes[color]
            # Explicit XY without explicit yaw reproduces the configured scene.
            yaw = changes.get("yaw_deg", cube["yaw_deg"] if "center_xy" in changes or not random["enabled"]
                              else float(rng.uniform(*random["yaw_range_deg"])))
            margin = cube["edge_m"]/2*(abs(math.sin(math.radians(yaw)))+abs(math.cos(math.radians(yaw))))
            bounds = [(random[key][0]+margin, random[key][1]-margin) for key in ("x_range_m", "y_range_m")]
            if any(high < low for low, high in bounds):
                raise ValueError("Sampling region cannot contain rotated cube footprint")
            xy = changes.get("center_xy", [float(rng.uniform(low, high)) for low, high in bounds]
                             if random["enabled"] else cube["center_xy"])
            result[color] = dict(center_xy=list(xy), yaw_deg=float(yaw), yaw_override=True)
        try:
            validate_layout(config, result)
            return result
        except ValueError as error:
            last_error = error
            if not random["enabled"] or all("center_xy" in overrides.get(color, {}) for color in COLORS):
                raise
    raise ValueError(f"Could not sample nonoverlapping cube positions after {random['max_attempts']} attempts: {last_error}")


def task_identity(config):
    scene = deepcopy(config["scene"])
    assets = {key: hashlib.sha256(Path(scene.pop(key)).read_bytes()).hexdigest() for key in ("robot_usd", "robot_urdf")}
    sources = {}
    for name in SOURCE_FILES:
        content = (ROOT/"src/pressb"/name).read_bytes()
        sources[name] = dict(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    semantics = {key: deepcopy(config[key]) for key in
        ("horizon", "control_hz", "physics_dt", "max_episode_seconds", "gamma", "settling_ticks", "smoothing",
         "model_inputs", "cubes", "randomization", "outcome", "reward", "terminate_on_success")}
    semantics.update(scene=scene, assets=assets, instruction=INSTRUCTION, outcome_version="contact_supported_cube_stack_v1")
    return dict(task_config_sha256=fingerprint(semantics), source_identity=sources,
        source_revision="sha256:"+fingerprint(sources), asset_sha256=assets,
        asset_scope="URDF and root robot USD; external USD references are not recursively hashed")
