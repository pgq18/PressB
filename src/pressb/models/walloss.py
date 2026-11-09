"""Frozen Wall-OSS adapter for the configurable inference node.

The public interface uses SI joint positions, while the original PAP checkpoint
uses degrees and full gripper width in centimetres. Conversion is explicit here;
normalisation remains the checkpoint's own normalisation. No Isaac dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from pressb.model_sources import activate_source, builtin_source_root, source_tree_sha256
from pressb.runtime.contract import decode_image
from pressb.runtime import image_transport
from pressb.runtime.image_transport import ValidatedObservation, walloss_resize_spec


PIPER_NAMES = [f"joint{i}" for i in range(1, 7)] + ["gripper_width"]
PIPER_UNITS = ["rad"] * 6 + ["m"]


def _hash_files(paths):
    """Content identity, independent of the machine's checkpoint directory."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        digest.update(path.name.encode() + b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _vector(value, size, name):
    result = np.asarray(value)
    if result.shape != (size,) or result.dtype.kind not in "fiu" or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite numeric vector of length {size}")
    return result.astype(np.float32)


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _flow_prefix_option(config):
    value = config.get("flow_prefix_cache", False)
    if type(value) is not bool:
        raise ValueError("flow_prefix_cache must be a boolean")
    return value


def _flow_prefix_mode(config):
    value = config.get("flow_prefix_cache_mode", "suffix")
    if value not in ("suffix", "full_attention"):
        raise ValueError("flow_prefix_cache_mode must be suffix or full_attention")
    return value


def _configure_cuda_budget(config, torch_module):
    value = config.get("cuda_memory_fraction")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value <= 1:
        raise ValueError("cuda_memory_fraction must be a finite number in (0, 1]")
    device = torch_module.device(config.get("device", "cuda:0"))
    if device.type != "cuda":
        raise ValueError("cuda_memory_fraction requires a CUDA device")
    torch_module.cuda.set_per_process_memory_fraction(float(value), device)
    return float(value)


def _image(value):
    return np.asarray(decode_image(value, require_rgb=True), dtype=np.uint8)


@dataclass
class WallContext:
    conditioning: dict
    baseline_noise: Any


class WallOSSBackend:
    """Model adapter; encodes once and reuses embeddings for any selected noise.

    Every call reaches the native policy as one model batch. The service owns
    microbatch sizing; the policy pads heterogeneous prompts and independently
    owns each encoded observation, so cached rows can be regrouped on decode.
    """

    def __init__(self, config: dict):
        import torch
        import yaml

        self.config = dict(config)
        self.flow_prefix_cache = _flow_prefix_option(config)
        self.flow_prefix_cache_mode = _flow_prefix_mode(config)
        self.cuda_memory_fraction = _configure_cuda_budget(config, torch)
        repo = activate_source("walloss", config.get("repo"), extension_dir=config.get("extension_dir"))
        from wall_x.serving.policy.wall_x_policy import WallXPolicy
        from wall_x.data.utils import load_norm_stats

        model_path = Path(config["model_path"]).expanduser().resolve()
        train_path = Path(config["train_config"]).expanduser().resolve()
        stats_path = Path(config.get("norm_stats", model_path / "norm_stats.json")).expanduser().resolve()
        for path in (model_path, train_path, stats_path):
            if not path.exists():
                raise FileNotFoundError(path)
        with train_path.open() as handle:
            train_config = yaml.safe_load(handle)
        train_config["data"]["model_type"] = train_config.get("model_type")
        train_config["norm_stats_path"] = str(stats_path)
        self.dataset_name = config.get("dataset_name", train_config["data"]["lerobot_config"]["repo_id"])
        self.names = list(config.get("names", PIPER_NAMES))
        self.units = list(config.get("units", PIPER_UNITS))
        if not self.names or len(self.names) != len(self.units) or len(set(self.names)) != len(self.names):
            raise ValueError("names/units must describe each unique action component")
        self.dimension = len(self.names)
        default_state_scale = [180 / np.pi] * 6 + [100.0]
        default_action_scale = [np.pi / 180] * 6 + [0.01]
        self.state_scale = _vector(config.get("state_to_model_scale", default_state_scale), self.dimension, "state_to_model_scale")
        self.action_scale = _vector(config.get("action_from_model_scale", default_action_scale), self.dimension, "action_from_model_scale")
        if np.any(self.state_scale == 0) or np.any(self.action_scale == 0):
            raise ValueError("Unit conversion scales must be nonzero")
        self.camera_mapping = dict(config.get("camera_mapping", {"up": "face_view", "wrist": "left_wrist_view"}))
        if not self.camera_mapping or len(set(self.camera_mapping.values())) != len(self.camera_mapping):
            raise ValueError("camera_mapping must map distinct simulation cameras to distinct model inputs")
        self.horizon = _positive_int(config.get("horizon", 32), "horizon")
        self.num_inference_timesteps = _positive_int(config.get("num_inference_timesteps", 10), "num_inference_timesteps")
        self.control_hz = float(config.get("control_hz", 30))
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self.model_id = config.get("model_id", model_path.name)
        self.image_preprocessing = walloss_resize_spec(config.get("input_image_resolution", 256))
        self.stats_path = str(stats_path)
        dtype_name = config.get("dtype", "bfloat16")
        if dtype_name not in ("bfloat16", "float16", "float32"):
            raise ValueError("dtype must be bfloat16, float16 or float32")
        stats = load_norm_stats(str(stats_path), self.dataset_name)
        self.state_min = _vector(stats["state"].min.cpu().numpy(), self.dimension, "state_min")
        self.state_delta = _vector(stats["state"].delta.cpu().numpy(), self.dimension, "state_delta")
        weights = list(model_path.glob("model*.safetensors")) or list(model_path.glob("pytorch_model*.bin"))
        if not weights:
            raise FileNotFoundError(f"No Hugging Face model weight files in {model_path}")
        self.weights_sha256 = _hash_files(weights)
        self.normalization_sha256 = hashlib.sha256(stats_path.read_bytes()).hexdigest()
        model_files = [path for path in model_path.iterdir() if path.suffix == ".json"]
        effective = {"horizon": self.horizon, "control_hz": self.control_hz,
                     "flow_prefix_cache": self.flow_prefix_cache,
                     "flow_prefix_cache_mode": self.flow_prefix_cache_mode,
                     "num_inference_timesteps": self.num_inference_timesteps,
                     "state_to_model_scale": self.state_scale.tolist(),
                     "action_from_model_scale": self.action_scale.tolist(),
                     "camera_mapping": self.camera_mapping, "dataset_name": self.dataset_name,
                     "input_image_resolution": config.get("input_image_resolution", 256), "dtype": dtype_name,
                     "checkpoint_config": _hash_files(model_files),
                     "train_config": hashlib.sha256(train_path.read_bytes()).hexdigest()}
        self.model_config_sha256 = hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()
        self.source_revision = _hash_files([
            Path(__file__), Path(image_transport.__file__), repo / "wall_x/model/flow_context.py",
            repo / "wall_x/model/flow_prefix.py",
            repo / "wall_x/model/vision_batch.py",
            repo / "wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py",
            repo / "csrc/window_index.cu",
            repo / "wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py",
            repo / "wall_x/serving/policy/wall_x_policy.py",
            repo / "wall_x/serving/policy/utils.py",
        ])
        if repo == builtin_source_root("walloss").resolve():
            self.source_revision = hashlib.sha256(json.dumps(dict(
                adapter_and_dependencies=self.source_revision,
                bundled_model_source=source_tree_sha256("walloss", repo)), sort_keys=True).encode()).hexdigest()
        self.policy = WallXPolicy(
            model_path=str(model_path), train_config=train_config,
            action_tokenizer_path=config.get("action_tokenizer_path"),
            action_dim=self.dimension, agent_pos_dim=self.dimension,
            pred_horizon=self.horizon, camera_key=list(self.camera_mapping.values()),
            device=config.get("device", "cuda:0"), dtype=config.get("dtype", "bfloat16"),
            predict_mode="diffusion", input_image_resolution=config.get("input_image_resolution", 256),
            flow_prefix_cache=self.flow_prefix_cache,
            flow_prefix_cache_mode=self.flow_prefix_cache_mode,
        )
        # Native policy historically selects bfloat16 regardless of its dtype
        # argument. Honour the generic node configuration explicitly, without
        # changing native callers' existing dtype behavior.
        self.policy.model.to(dtype=getattr(torch, dtype_name))
        self.policy.model.eval().requires_grad_(False)
        self.noise_dim = self.policy.fixed_action_dim
        self.feature_dim = int(self.policy.model.config.hidden_size)

    def describe(self):
        return {
            "schema_version": 2, "node_type": "inference", "model_id": self.model_id,
            "capabilities": ["encode", "decode", "infer", "external_initial_noise", "features", "context_cache"],
            "observation": {"state": {"names": self.names, "units": self.units}, "cameras": list(self.camera_mapping)},
            "action": {"representation": "joint_position", "frame": "robot_base", "names": self.names,
                       "units": self.units, "horizon": self.horizon, "control_hz": self.control_hz},
            "noise": {"shape": [self.horizon, self.noise_dim]},
            "feature_dim": self.feature_dim,
            "feature_kind": "mean_multimodal_embedding_excluding_action_and_padding",
            "cache_kind": "multimodal_embeddings_and_rope_indices",
            "transformer_kv_cache": False,
            "diffusion_prefix_reuse": getattr(self, "flow_prefix_cache", False),
            "diffusion_prefix_kv_reuse": (getattr(self, "flow_prefix_cache", False)
                and getattr(self, "flow_prefix_cache_mode", "suffix") == "suffix"),
            "diffusion_prefix_cache_scope": "within_one_decode",
            "diffusion_prefix_cache_mode": getattr(self, "flow_prefix_cache_mode", "suffix"),
            "batching": {"native_model_batching": True},
            "frozen": True, "norm_stats": self.stats_path,
            "num_inference_timesteps": self.num_inference_timesteps,
            "cuda_memory_fraction": getattr(self, "cuda_memory_fraction", None),
            "model_inputs": {"dataset_name": self.dataset_name},
            "weights_sha256": self.weights_sha256,
            "normalization_sha256": self.normalization_sha256,
            "model_config_sha256": self.model_config_sha256,
            "source_revision": self.source_revision,
            "image_encodings": ["png_base64", "png_bytes"],
            "image_mode": "RGB",
            "image_preprocessing": getattr(self, "image_preprocessing", None),
        }

    def create_action_expert(self, config=None):
        from .flow_experts import create_walloss_expert
        return create_walloss_expert(self, config)

    def _prepare_observation(self, observation):
        if isinstance(observation, ValidatedObservation) and observation.prepared_walloss is not None:
            return observation.prepared_walloss
        model_inputs = observation.get("model_inputs", {})
        if not isinstance(model_inputs, dict) or set(model_inputs) - {"dataset_name"}:
            raise ValueError("Wall-OSS model_inputs supports only dataset_name")
        if model_inputs.get("dataset_name", self.dataset_name) != self.dataset_name:
            raise ValueError("dataset_name must match the configured checkpoint normalizer")
        state = _vector(observation["state"], self.dimension, "observation.state") * self.state_scale
        delta = np.where(self.state_delta == 0, 1, self.state_delta)
        state = np.clip(2 * (state - self.state_min) / delta - 1, -1, 1)
        prompt = observation["task"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("task must be a nonempty instruction")
        images = ({source: np.asarray(image, dtype=np.uint8)
                   for source, image in observation.decoded_images.items()}
                  if isinstance(observation, ValidatedObservation) else
                  {source: _image(observation["images"][source]) for source in self.camera_mapping})
        prepared = {
            "state": state, "prompt": prompt, "dataset_names": [self.dataset_name],
            **{target: images[source] for source, target in self.camera_mapping.items()},
        }
        if isinstance(observation, ValidatedObservation):
            observation.prepared_walloss = prepared
        return prepared

    def encode_batch(self, observations, seeds):
        import torch
        if len(observations) != len(seeds):
            raise ValueError("Each observation requires its own seed")
        if not observations:
            return []
        # Validate all rows before starting GPU computation. The policy does
        # camera/token preparation and multimodal encoding on the whole batch.
        prepared = [self._prepare_observation(observation) for observation in observations]
        conditionings = self.policy.encode_batch(
            prepared, num_inference_timesteps=self.num_inference_timesteps,
        )
        if len(conditionings) != len(observations):
            raise ValueError("Wall-OSS returned an incorrect encoder batch size")
        result = []
        for conditioning, seed in zip(conditionings, seeds):
            reference = conditioning["inputs"]["inputs_embeds"]
            # One generator per observation is deliberate: a seed must produce
            # the same baseline noise regardless of batch size or row order.
            generator = torch.Generator(device=reference.device).manual_seed(int(seed))
            baseline = torch.randn((1, self.horizon, self.noise_dim), device=reference.device,
                                   dtype=reference.dtype, generator=generator)
            feature = conditioning["feature"][0].detach().float().cpu().numpy().copy()
            if feature.shape != (self.feature_dim,) or not np.isfinite(feature).all():
                raise ValueError("Wall-OSS produced invalid observation features")
            result.append((WallContext(conditioning, baseline), feature))
        return result

    def close(self):
        self.policy = None

    def validate_observations(self, observations):
        """Check the whole request before the service starts model computation."""
        for observation in observations:
            self._prepare_observation(observation)

    def decode_batch(self, contexts, initial_noises):
        import torch
        if len(contexts) != len(initial_noises):
            raise ValueError("Each context requires its own optional noise")
        if not contexts:
            return []
        noises = []
        for context, noise in zip(contexts, initial_noises):
            if noise is None:
                noise = context.baseline_noise
            else:
                value = np.asarray(noise)
                if value.shape != (self.horizon, self.noise_dim) or value.dtype.kind != "f" or not np.isfinite(value).all():
                    raise ValueError(f"initial_noise must be finite float [{self.horizon}, {self.noise_dim}]")
                noise = torch.as_tensor(value, device=context.baseline_noise.device,
                                        dtype=context.baseline_noise.dtype).unsqueeze(0)
            noises.append(noise)
        output = self.policy.decode_batch(
            [context.conditioning for context in contexts], initial_noise=torch.cat(noises, dim=0),
        )["action"]
        actions = np.asarray(output, dtype=np.float32)
        if actions.shape != (len(contexts), self.horizon, self.dimension) or not np.isfinite(actions).all():
            raise ValueError("Wall-OSS produced invalid batched actions")
        actions = actions * self.action_scale
        return [action.copy() for action in actions]


def create_backend(config: dict):
    return WallOSSBackend(config)
