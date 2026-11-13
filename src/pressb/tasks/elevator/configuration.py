"""CPU-only validation for the frozen elevator scene and simulation settings."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_INPUT = ROOT / "outputs/rtx5090_eval_comparison/input"
_DEFAULTS = dict(gpu=0, num_envs=2, max_seconds=15., smoothing_window=3, single_gamma=.99,
    response_cache_size=2, camera_resolution=224, render_subframes=1,
    light_settle_captures=1, light_settle_subframes=2, ik_iterations=16,
    log_actions=False, png_workers=4, cpu_threads=8, color_response_workers=0, gpu_dynamics=False)
_PATHS = dict(config=_INPUT / "config.json", snapshot=_INPUT / "scene.usda",
              dataset=_INPUT / "dataset_metadata", asset_bundle=_INPUT / "assets")
_ADAPTER_KEYS = {"floors", "panel_randomization", "run_id", "expected_metadata"}


def _path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(config):
    """Validate the launch and frozen scene before importing Isaac or Kit."""
    allowed = set(_DEFAULTS) | set(_PATHS) | _ADAPTER_KEYS | {"output"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError(f"Native elevator configuration supports {sorted(allowed)}")
    values = {**_DEFAULTS, **copy.deepcopy(config)}
    for key in ("gpu", "num_envs", "response_cache_size", "render_subframes",
                "light_settle_captures", "light_settle_subframes", "ik_iterations",
                "png_workers", "cpu_threads", "color_response_workers"):
        minimum = 0 if key in ("gpu", "light_settle_captures", "color_response_workers") else 1
        if type(values[key]) is not int or values[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if type(values["smoothing_window"]) is not int or values["smoothing_window"] not in (1, 3, 5, 7, 9, 11):
        raise ValueError("smoothing_window must be one of 1,3,5,7,9,11")
    if type(values["camera_resolution"]) is not int or values["camera_resolution"] not in (224, 640):
        raise ValueError("camera_resolution must be 224 or 640")
    for key in ("max_seconds", "single_gamma"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    if not 0 < values["max_seconds"] <= 120 or abs(values["max_seconds"] * 120 - round(values["max_seconds"] * 120)) > 1e-7:
        raise ValueError("max_seconds must be in (0,120] and on a 120 Hz tick")
    if not 0 < values["single_gamma"] <= 1:
        raise ValueError("single_gamma must be in (0,1]")
    for key in ("log_actions", "gpu_dynamics"):
        if type(values[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key, default in _PATHS.items():
        value = values.get(key, default)
        values[key] = None if key == "asset_bundle" and value is None else _path(value)
    if not isinstance(values.get("output"), (str, Path)) or not str(values["output"]):
        raise ValueError("Native elevator requires a new output directory")
    values["output"] = _path(values["output"])
    values["native_log_path"] = values["output"].with_name(values["output"].name + ".kit.log")
    for path in (values["output"], values["native_log_path"]):
        if path.exists():
            raise FileExistsError(f"Native elevator output must be new: {path}")
    scene = json.loads(values["config"].read_text())
    collection = json.loads((values["dataset"] / "meta/collection_metadata.json").read_text())
    scene_hash = hashlib.sha256(values["snapshot"].read_bytes()).hexdigest()
    if scene != collection["config"] or scene_hash != collection["scene_sha256"]:
        raise ValueError("Config and scene must match the frozen training collection")
    bundle = values["asset_bundle"]
    if bundle is not None and not (bundle / "asset_bundle.json").is_file():
        raise FileNotFoundError("asset_bundle.json is missing from the specified bundle")
    # Apply the bridge's cheap configuration validation before opening Kit.
    floors = values.get("floors", list(range(24, 36)))
    if (not isinstance(floors, (list, tuple)) or not floors
            or any(type(f) is not int or not 24 <= f <= 35 for f in floors)
            or len(set(floors)) != len(floors)):
        raise ValueError("floors must be distinct integers in 24..35")
    if type(values.get("panel_randomization", True)) is not bool:
        raise ValueError("panel_randomization must be boolean")
    if "run_id" in values and (not isinstance(values["run_id"], str)
            or not values["run_id"].strip() or len(values["run_id"]) > 256):
        raise ValueError("run_id must be a nonempty string of at most 256 characters")
    if not isinstance(values.get("expected_metadata", {}), dict):
        raise ValueError("expected_metadata must be a mapping")
    return values
