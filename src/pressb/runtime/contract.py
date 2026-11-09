"""Typed-by-metadata boundaries between frozen inference, environments and RL.

All wire actions are absolute executable commands in the declared units.
Latent noise dimensions are deliberately independent of actuator dimensions.
The surrounding legacy transport envelope remains version 1; schema_version=2
identifies this task-independent API inside it.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
from io import BytesIO
import json
import math

import numpy as np
from PIL import Image

from . import SCHEMA_VERSION


def finite_array(value, shape=None, name="array", dtype=np.float32):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(value, dtype=dtype)
    if (shape is not None and result.shape != tuple(shape)) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}; got {result.shape}")
    return result


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def normalize_env_seeds(env_ids, env_seeds):
    """Validate optional per-slot reset seeds without changing seed semantics.

    JSON object keys arrive as strings, while direct Python callers may use
    integer keys. Only their canonical forms are accepted, and aliases must not
    name the same slot twice. ``None`` retains the legacy shared-seed reset.
    Callers still validate environment bounds and the required shared seed.
    """
    if env_seeds is None:
        return None
    if (not isinstance(env_ids, list) or not env_ids
            or any(type(i) is not int or i < 0 for i in env_ids)
            or len(set(env_ids)) != len(env_ids)):
        raise ValueError("env_ids must contain distinct nonnegative integer indices")
    if not isinstance(env_seeds, dict):
        raise ValueError("env_seeds must be an object mapping environment ids to seeds")
    normalized = {}
    for key, seed in env_seeds.items():
        if type(key) is int and key >= 0:
            env_id = key
        elif (isinstance(key, str) and key.isascii() and key.isdecimal()
              and str(int(key)) == key):
            env_id = int(key)
        else:
            raise ValueError("env_seeds keys must be nonnegative integer ids or canonical decimal strings")
        if env_id in normalized:
            raise ValueError("env_seeds contains duplicate normalized environment ids")
        if type(seed) is not int or seed < 0:
            raise ValueError("env_seeds values must be nonnegative integer seeds")
        normalized[env_id] = seed
    if set(normalized) != set(env_ids):
        raise ValueError("env_seeds must exactly cover env_ids")
    return normalized


def _named_vector(spec, name):
    if not isinstance(spec, dict):
        raise ValueError(f"{name} must declare names and units")
    names, units = spec.get("names"), spec.get("units")
    if (not isinstance(names, list) or not names
            or not all(isinstance(x, str) and x for x in names)
            or len(set(names)) != len(names)):
        raise ValueError(f"{name}.names must be unique nonempty strings")
    if (not isinstance(units, list) or len(units) != len(names)
            or not all(isinstance(x, str) and x for x in units)):
        raise ValueError(f"{name}.units must match its names")


def validate_metadata(metadata, node_type=None):
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Node must explicitly advertise schema_version=2")
    kind = metadata.get("node_type")
    if kind not in ("inference", "simulation") or (node_type is not None and kind != node_type):
        raise ValueError(f"Expected {node_type or 'inference/simulation'} node")
    obs = metadata.get("observation", {})
    _named_vector(obs.get("state"), "observation.state")
    cameras = obs.get("cameras")
    if (not isinstance(cameras, list) or not all(isinstance(x, str) and x for x in cameras)
            or len(set(cameras)) != len(cameras)):
        raise ValueError("observation.cameras must be distinct camera names")
    action = metadata.get("action", {})
    _named_vector(action, "action")
    _positive_int(action.get("horizon"), "action.horizon")
    for name in ("representation", "frame"):
        if not isinstance(action.get(name), str) or not action[name]:
            raise ValueError(f"action.{name} must be declared")
    rate = action.get("control_hz")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("action.control_hz must be finite and positive")
    for key in ("lower", "upper"):
        if key in action:
            finite_array(action[key], (len(action["names"]),), f"action.{key}")
    if "lower" in action and "upper" in action and np.any(np.array(action["lower"]) >= action["upper"]):
        raise ValueError("action.lower must be below action.upper")
    if kind == "inference":
        _positive_int(metadata.get("feature_dim"), "feature_dim")
        noise_shape = metadata.get("noise", {}).get("shape")
        if not isinstance(noise_shape, list) or len(noise_shape) != 2:
            raise ValueError("noise.shape must be [horizon, latent_dim]")
        for dim in noise_shape:
            _positive_int(dim, "noise.shape dimension")
        if noise_shape[0] != action["horizon"]:
            raise ValueError("Noise and action horizons must agree")
        if not isinstance(metadata.get("model_id"), str) or not metadata["model_id"]:
            raise ValueError("Inference must declare model_id")
        caps = metadata.get("capabilities", [])
        if not isinstance(caps, list) or not all(isinstance(c, str) for c in caps):
            raise ValueError("capabilities must be a list of strings")
    else:
        _positive_int(metadata.get("num_envs"), "num_envs")
        if not isinstance(metadata.get("task_id"), str) or not metadata["task_id"]:
            raise ValueError("Simulation must declare task_id")
        for key in ("control_state_dim", "task_features_dim"):
            if type(metadata.get(key)) is not int or metadata[key] < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        discount = metadata.get("discount", {})
        gamma, hz = discount.get("gamma"), discount.get("timebase_hz")
        if (isinstance(gamma, bool) or not isinstance(gamma, (int, float))
                or not math.isfinite(gamma) or not 0 < gamma <= 1):
            raise ValueError("discount.gamma must be in (0,1]")
        if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not math.isfinite(hz) or hz <= 0:
            raise ValueError("discount.timebase_hz must be positive and finite")
        if metadata.get("reset_mode") != "indexed":
            raise ValueError("Simulation must support independent indexed resets")
        if "equal_execution_horizon" in metadata and type(metadata["equal_execution_horizon"]) is not bool:
            raise ValueError("equal_execution_horizon must be a boolean")
        if "capabilities" in metadata and (not isinstance(metadata["capabilities"], list)
                or not all(isinstance(value, str) for value in metadata["capabilities"])):
            raise ValueError("capabilities must be a list of strings")
        if "terminal_observation_modes" in metadata:
            modes = metadata["terminal_observation_modes"]
            if (not isinstance(modes, list) or "full" not in modes
                    or any(value not in ("full", "state_only_if_no_bootstrap") for value in modes)):
                raise ValueError("Invalid terminal_observation_modes")
    return metadata


def validate_observation(observation, spec, *, allow_missing_images=False):
    if not isinstance(observation, dict):
        raise ValueError("Observation must be an object")
    finite_array(observation.get("state"), (len(spec["state"]["names"]),), "observation.state")
    if not isinstance(observation.get("task"), str) or not observation["task"].strip():
        raise ValueError("Observation must include a nonempty task instruction")
    images = observation.get("images")
    if not isinstance(images, dict):
        raise ValueError("Observation images must be an object")
    if allow_missing_images and images:
        raise ValueError("An omitted terminal image observation must contain an empty images object")
    if not allow_missing_images and not set(spec["cameras"]).issubset(images):
        raise ValueError("Observation is missing a declared camera")
    for name in (() if allow_missing_images else spec["cameras"]):
        value = images[name]
        if not isinstance(value, dict):
            raise ValueError(f"Camera {name} requires PNG image data")
        encoding, data = value.get("encoding"), value.get("data")
        if not ((encoding == "png_base64" and isinstance(data, str) and data)
                or (encoding == "png_bytes" and isinstance(data, bytes) and data)):
            raise ValueError(f"Camera {name} requires png_base64 or png_bytes image data")
    if not isinstance(observation.get("model_inputs", {}), dict):
        raise ValueError("model_inputs must be an object")
    return observation


def encode_image(image):
    image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image, dtype=np.uint8))
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return {"encoding": "png_base64", "data": base64.b64encode(buffer.getvalue()).decode("ascii")}


def decode_image(value, *, require_rgb=False):
    if not isinstance(value, dict) or value.get("encoding") not in ("png_base64", "png_bytes"):
        raise ValueError("Expected a png_base64 or png_bytes image")
    try:
        if value["encoding"] == "png_bytes":
            data = value["data"]
            if not isinstance(data, bytes):
                raise ValueError("png_bytes data must be bytes")
        else:
            data = base64.b64decode(value["data"], validate=True)
        if len(data) > 16 * 1024 * 1024:
            raise ValueError("Image exceeds 16 MiB")
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG" or image.width * image.height > 4096 * 4096:
                raise ValueError("Expected PNG image of at most 4096² pixels")
            if require_rgb and image.mode != "RGB":
                raise ValueError("Expected an RGB PNG image")
            return image.convert("RGB")
    except (KeyError, binascii.Error, OSError) as error:
        raise ValueError("Invalid PNG image") from error


def validate_actions(actions, spec, allow_prefix=False):
    array = finite_array(actions, name="actions")
    horizon, width = spec["horizon"], len(spec["names"])
    if (array.ndim != 2 or array.shape[1] != width
            or not (1 <= array.shape[0] <= horizon if allow_prefix else array.shape[0] == horizon)):
        raise ValueError(f"Actions must be [{'1..' if allow_prefix else ''}{horizon},{width}]")
    if "lower" in spec and np.any(array < np.asarray(spec["lower"]) - 1e-6):
        raise ValueError("Action falls below the executable lower limit")
    if "upper" in spec and np.any(array > np.asarray(spec["upper"]) + 1e-6):
        raise ValueError("Action exceeds the executable upper limit")
    return array


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def negotiate(inference_metadata, simulation_metadata, method, execution_horizon=None, *,
              terminal_observation_mode="full", fused_step_reset=False):
    inference = validate_metadata(inference_metadata, "inference")
    simulation = validate_metadata(simulation_metadata, "simulation")
    # Bind producer-side model resizing before the first reset. Pixels already
    # resized for another model cannot be recovered by the inference receiver.
    if simulation.get("image_preprocessing") is not None:
        from .image_transport import validate_image_preprocessing
        producer_spec = validate_image_preprocessing(simulation["image_preprocessing"])
        if producer_spec != inference.get("image_preprocessing"):
            raise ValueError("Simulation image_preprocessing does not match inference image_preprocessing")
    if method not in ("dsrl", "initial_noise", "action_residual", "base", "dppo"):
        raise ValueError("Unknown training method")
    validate_terminal_observation_mode(terminal_observation_mode, simulation)
    if type(fused_step_reset) is not bool:
        raise ValueError("fused_step_reset must be boolean")
    if fused_step_reset and (terminal_observation_mode != "state_only_if_no_bootstrap"
                            or "step_and_reset" not in simulation.get("capabilities", [])):
        raise ValueError("fused_step_reset requires state-only terminals and simulation step_and_reset support")
    required = {"encode", "decode", "features", "context_cache"}
    if method in ("dsrl", "initial_noise"):
        required.add("external_initial_noise")
    if method == "dppo":
        required.update(("sample_chain", "action_expert_update"))
    if not required.issubset(inference.get("capabilities", [])):
        raise ValueError(f"Inference missing capabilities: {sorted(required - set(inference.get('capabilities', [])))}")
    ia, sa = inference["action"], simulation["action"]
    for key in ("representation", "frame", "names", "units", "control_hz"):
        if ia[key] != sa[key]:
            raise ValueError(f"Inference/simulation action.{key} mismatch; configure an explicit adapter")
    io, so = inference["observation"], simulation["observation"]
    for key in ("names", "units"):
        if io["state"][key] != so["state"][key]:
            raise ValueError(f"Inference/simulation observation.state.{key} mismatch")
    if not set(io["cameras"]).issubset(so["cameras"]):
        raise ValueError("Simulation does not supply the model's required cameras")
    prefix = ia["horizon"] if execution_horizon is None else _positive_int(execution_horizon, "execution_horizon")
    if prefix > min(ia["horizon"], sa["horizon"]):
        raise ValueError("Execution horizon exceeds model output or simulation capacity")
    if "execution_horizons" in simulation and prefix not in simulation["execution_horizons"]:
        raise ValueError("Simulation adapter does not support this execution horizon")
    state_dim = len(io["state"]["names"])
    result = dict(schema_version=SCHEMA_VERSION, action_horizon=ia["horizon"],
                  action_dim=len(ia["names"]), noise_shape=inference["noise"]["shape"],
                  feature_dim=inference["feature_dim"], state_dim=state_dim,
                  control_state_dim=simulation["control_state_dim"],
                  task_features_dim=simulation["task_features_dim"], execution_horizon=prefix,
                  action=ia, discount=simulation["discount"], num_envs=simulation["num_envs"],
                  observation=io, method="dsrl" if method == "initial_noise" else method)
    result["obs_dim"] = sum(result[key] for key in ("feature_dim", "state_dim", "control_state_dim", "task_features_dim"))
    # Keep volatile cache occupancy / runtime counters out of resume identity.
    result["identity"] = {
        "inference": {k: inference[k] for k in ("model_id", "checkpoint_sha256", "weights_sha256", "source_revision", "normalization_sha256", "model_config_sha256", "image_preprocessing") if k in inference},
        "simulation": {k: simulation[k] for k in ("task_id", "task_config_sha256", "source_revision", "smoothing", "image_preprocessing") if k in simulation},
    }
    if method == "dppo":
        expert = inference.get("action_expert", {})
        if not isinstance(expert.get("spec"), dict):
            raise ValueError("DPPO requires a declared action expert sampler specification")
        result["identity"]["action_expert"] = {key: expert[key] for key in ("spec", "config") if key in expert}
    if terminal_observation_mode != "full":
        result["terminal_observation_mode"] = terminal_observation_mode
    if fused_step_reset:
        result["fused_step_reset"] = True
    result["fingerprint"] = fingerprint({k: v for k, v in result.items() if k != "num_envs"})
    return result


def validate_terminal_observation_mode(mode, metadata):
    """An explicit per-step agreement; old simulation backends remain full-only."""
    if mode not in ("full", "state_only_if_no_bootstrap"):
        raise ValueError("Unknown terminal_observation_mode")
    if mode != "full" and (
            "terminal_state_only" not in metadata.get("capabilities", [])
            or mode not in metadata.get("terminal_observation_modes", [])):
        raise ValueError("Simulation does not advertise terminal_state_only support")
    return mode
