"""LeRobot π0.5 inference through the PressB model boundary."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MethodType

import numpy as np

from pressb.runtime.contract import decode_image, finite_array, fingerprint, validate_metadata, validate_observation
from pressb.runtime.image_transport import ValidatedObservation


IO_FILENAME = "pressb_pi05.json"
STATE_KEY = "observation.state"
IMAGE_PREFIX = "observation.images."
PREPROCESSOR_FILENAME = "policy_preprocessor.json"
POSTPROCESSOR_FILENAME = "policy_postprocessor.json"


def _read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initialize_pi05_device(device):
    """Register the native NPU backend only when requested, and reject unavailable NPUs."""
    import torch

    if str(device).split(":", 1)[0] == "npu":
        try:
            import torch_npu  # noqa: F401  # Registers torch.npu and native ATen kernels.
        except ImportError as exc:
            raise RuntimeError("π0.5 NPU execution requires a matching torch_npu installation") from exc
        if not torch.npu.is_available():
            raise RuntimeError("π0.5 NPU execution requested, but no Ascend device is available")
        result = torch.device(device)
        if result.index is not None and result.index >= torch.npu.device_count():
            raise ValueError(f"π0.5 NPU device {result} is unavailable")
        return result
    return torch.device(device)


def parse_pi05_config(options):
    """Decode LeRobot options without its CUDA/CPU-only device auto-selection on NPU."""
    import draccus
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    options = deepcopy(options)
    requested = options.get("device")
    is_npu = str(requested).split(":", 1)[0] == "npu"
    if is_npu:
        requested = str(initialize_pi05_device(requested))
        options["device"] = "cpu"
    config = draccus.decode(PI05Config, options)
    if is_npu:
        config.device = requested
        config.use_amp = bool(options.get("use_amp", False))
    return config


def load_pi05_config(checkpoint, *, device=None):
    """Read a local LeRobot policy configuration with an optional device override."""
    # Upstream from_pretrained instantiates the saved configuration *before*
    # applying device overrides. Decode once so an NPU checkpoint is portable
    # to CUDA/CPU even when torch_npu is not installed on the inference host.
    options = _read_json(Path(checkpoint).expanduser() / "config.json")
    if options.pop("type", None) != "pi05":
        raise ValueError("Expected a LeRobot pi05 policy config")
    if device is not None:
        options["device"] = str(device)
    return parse_pi05_config(options)


def _embed_suffix_npu(self, noisy_actions, timestep):
    """LeRobot π0.5 suffix embedding with FP32 phases (Ascend has no FP64 tensors).

    Bound to an individual NPU policy, so CUDA/CPU models and upstream modules
    retain their own implementation. Parameter names and checkpoint format stay
    identical to LeRobot's pinned PI05Pytorch.embed_suffix.
    """
    import torch
    from torch.nn import functional as F

    dimension = self.action_in_proj.out_features
    if dimension % 2 or timestep.ndim != 1:
        raise ValueError("π0.5 timestep embedding needs an even dimension and a batch vector")
    fraction = torch.linspace(0., 1., dimension // 2, dtype=torch.float32, device=timestep.device)
    period = self.config.min_period * (self.config.max_period / self.config.min_period) ** fraction
    phase = (2 * math.pi / period)[None, :] * timestep.float()[:, None]
    time_emb = torch.cat([phase.sin(), phase.cos()], dim=1).to(timestep.dtype)

    action_emb = self._apply_checkpoint(self.action_in_proj, noisy_actions)

    def time_mlp_func(value):
        return F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(value))))

    conditioning = self._apply_checkpoint(time_mlp_func, time_emb)
    pad_masks = torch.ones(action_emb.shape[:2], dtype=torch.bool, device=timestep.device)
    att_masks = torch.tensor([1] + [0] * (self.config.chunk_size - 1),
                             dtype=action_emb.dtype, device=action_emb.device)
    return action_emb, pad_masks, att_masks[None, :].expand(len(action_emb), -1), conditioning


def create_pi05_policy(config):
    """Construct a native LeRobot policy with scoped accelerator compatibility."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    is_npu = str(getattr(config, "device", "cpu")).split(":", 1)[0] == "npu"
    if is_npu:
        initialize_pi05_device(config.device)
    policy = PI05Policy(config)
    if is_npu:
        policy.model.embed_suffix = MethodType(_embed_suffix_npu, policy.model)
    return policy


def load_pi05_policy(checkpoint, config=None, *, device=None, pretrained_base=False):
    """Restore all tensors from a native checkpoint or an upstream OpenPI conversion."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from safetensors.torch import load_file, load_model

    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    weight_path = checkpoint / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"Missing π0.5 weights: {weight_path}")
    if config is None:
        config = load_pi05_config(checkpoint, device=device)
    elif device is not None:
        config = deepcopy(config)
        config.device = str(device)
    if not isinstance(config, PI05Policy.config_class):
        raise ValueError("Expected a LeRobot pi05 policy config")
    policy = create_pi05_policy(config)
    if pretrained_base:
        # LeRobot's converted OpenPI checkpoints can omit the policy prefix.
        # Apply the upstream remap before a strict load, since the native
        # PI05Policy.from_pretrained can return a model after a load failure.
        weights = policy._fix_pytorch_state_dict_keys(load_file(weight_path), config)
        remapped = {key if key.startswith("model.") else f"model.{key}": value for key, value in weights.items()}
        if len(remapped) != len(weights):
            raise ValueError("Duplicate parameter names after π0.5 pretrained-key remapping")
        # Official safetensors can store a tied parameter only once. Restore
        # missing aliases only where the constructed model actually ties them.
        aliases = {}
        for name, parameter in policy.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(parameter), []).append(name)
        for names in aliases.values():
            if len(names) > 1:
                saved = [name for name in names if name in remapped]
                if saved:
                    reference = remapped[saved[0]]
                    if any(not reference.equal(remapped[name]) for name in saved[1:]):
                        raise ValueError(f"Inconsistent π0.5 tied parameter values: {names}")
                    for name in names:
                        remapped.setdefault(name, reference)
        policy.load_state_dict(remapped, strict=True)
    else:
        load_model(policy, weight_path, strict=True)
    return policy


def _load_processors(checkpoint, config, device):
    from lerobot.configs.types import FeatureType, NormalizationMode
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import processor_pi05  # noqa: F401  # Register the native tokenizer step.
    from lerobot.processor import NormalizerProcessorStep

    paths = [checkpoint / name for name in (PREPROCESSOR_FILENAME, POSTPROCESSOR_FILENAME)]
    processor_files = {}
    for path in paths:
        pipeline = _read_json(path)
        processor_files[path.name] = _sha256(path)
        for step in pipeline["steps"]:
            if name := step.get("state_file"):
                processor_files[name] = _sha256(checkpoint / name)
    overrides = {"device_processor": {"device": str(device)}}
    tokenizer = checkpoint / "tokenizer"
    if tokenizer.is_dir():
        overrides["tokenizer_processor"] = {"tokenizer_name": str(tokenizer)}
        processor_files.update({str(path.relative_to(checkpoint)): _sha256(path)
                                for path in sorted(tokenizer.rglob("*")) if path.is_file()})
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=str(checkpoint), preprocessor_overrides=overrides,
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    for step in preprocessor.steps:
        if (isinstance(step, NormalizerProcessorStep)
                and step.norm_map.get(FeatureType.VISUAL, NormalizationMode.IDENTITY) != NormalizationMode.IDENTITY):
            raise ValueError("π0.5 batching expects the native identity visual normalization")
    return preprocessor, postprocessor, fingerprint(processor_files)


def _wire_contract(value, config, model_id):
    allowed = {"format", "checkpoint_step", "lerobot_version", "observation", "action", "camera_keys",
               "state_scale", "action_scale"}
    if not isinstance(value, dict) or set(value) - allowed or not {"observation", "action"} <= set(value):
        raise ValueError(f"π0.5 wire metadata requires observation/action and supports {sorted(allowed)}")
    if value.get("format", "pressb.pi05.v1") != "pressb.pi05.v1":
        raise ValueError("Unknown π0.5 wire metadata format")
    if "checkpoint_step" in value and (type(value["checkpoint_step"]) is not int or value["checkpoint_step"] < 0):
        raise ValueError("checkpoint_step must be a nonnegative integer")
    if "lerobot_version" in value and (
            not isinstance(value["lerobot_version"], str) or not value["lerobot_version"]):
        raise ValueError("lerobot_version must be a nonempty string")
    observation, action = deepcopy(value["observation"]), deepcopy(value["action"])
    if not isinstance(observation, dict):
        raise ValueError("π0.5 wire observation metadata must be an object")
    if not isinstance(action, dict) or action.get("horizon", config.chunk_size) != config.chunk_size:
        raise ValueError("π0.5 wire action horizon must equal the policy chunk_size")
    action["horizon"] = config.chunk_size
    state_dim = config.robot_state_feature.shape[0]
    action_dim = config.action_feature.shape[0]
    metadata = validate_metadata(dict(
        schema_version=2, node_type="inference", model_id=model_id, observation=observation, action=action,
        noise=dict(shape=[config.chunk_size, config.max_action_dim], semantics="flow_initial_state"),
        feature_dim=state_dim, capabilities=["encode", "decode", "infer", "features", "context_cache",
                                              "external_initial_noise"],
    ), "inference")
    if "checkpoint_step" in value:
        metadata["checkpoint_step"] = value["checkpoint_step"]
    if len(observation["state"]["names"]) != state_dim or len(action["names"]) != action_dim:
        raise ValueError("π0.5 wire state/action dimensions do not match the checkpoint features")
    cameras = observation["cameras"]
    if not cameras:
        raise ValueError("π0.5 requires at least one camera")
    keys = value.get("camera_keys", {camera: IMAGE_PREFIX + camera for camera in cameras})
    if (not isinstance(keys, dict) or set(keys) != set(cameras)
            or any(not isinstance(key, str) or not key.startswith(IMAGE_PREFIX) for key in keys.values())
            or len(set(keys.values())) != len(cameras)):
        raise ValueError("camera_keys must map each wire camera to a distinct LeRobot observation.images key")
    real_images = {key for key in config.image_features if not key.startswith(IMAGE_PREFIX + "empty_camera_")}
    if set(keys.values()) != real_images:
        raise ValueError("π0.5 camera_keys must cover the policy's nonempty image features")
    scales = []
    for key, dim in (("state_scale", state_dim), ("action_scale", action_dim)):
        scale = finite_array(value.get(key, [1.] * dim), (dim,), name=key).copy()
        if (scale == 0).any():
            raise ValueError(f"{key} entries must be nonzero")
        scales.append(scale)
    return metadata, dict(keys), *scales


@dataclass
class Pi05Context:
    processed: dict
    seed: int


class Pi05Backend:
    def __init__(self, config):
        import torch

        allowed = {"checkpoint", "device", "io", "model_id", "num_inference_steps", "pretrained_base"}
        if not isinstance(config, dict) or set(config) - allowed or not config.get("checkpoint"):
            raise ValueError(f"π0.5 requires a local checkpoint and supports {sorted(allowed)}")
        if type(config.get("pretrained_base", False)) is not bool:
            raise ValueError("pretrained_base must be a boolean")
        checkpoint = Path(config["checkpoint"]).expanduser().resolve(strict=True)
        self.device = initialize_pi05_device(config.get("device", "cuda:0"))
        if self.device.type not in ("cpu", "cuda", "npu"):
            raise ValueError("π0.5 inference device must be CPU, CUDA or NPU")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("π0.5 inference requested CUDA but it is unavailable")
        from lerobot import __version__ as lerobot_version

        policy_config = load_pi05_config(checkpoint, device=self.device)
        policy_config.validate_features()
        if policy_config.rtc_config is not None and policy_config.rtc_config.enabled:
            raise ValueError("π0.5 PressB inference requires RTC to be disabled")
        if policy_config.robot_state_feature is None or policy_config.action_feature is None:
            raise ValueError("π0.5 checkpoint requires state and action features")
        if (len(policy_config.robot_state_feature.shape) != 1 or len(policy_config.action_feature.shape) != 1
                or policy_config.robot_state_feature.shape[0] > policy_config.max_state_dim
                or policy_config.action_feature.shape[0] > policy_config.max_action_dim):
            raise ValueError("π0.5 expects vector state/action features within the model's padded dimensions")
        if "num_inference_steps" in config:
            steps = config["num_inference_steps"]
            if type(steps) is not int or steps < 1:
                raise ValueError("num_inference_steps must be a positive integer")
            policy_config.num_inference_steps = steps
        digest = _sha256(checkpoint / "model.safetensors")
        contract = config.get("io")
        if contract is None:
            contract = _read_json(checkpoint / IO_FILENAME)
        self._metadata, self.camera_keys, self.state_scale, self.action_scale = _wire_contract(
            contract, policy_config, config.get("model_id", "pi05:" + digest),
        )
        self.preprocessor, self.postprocessor, processor_digest = _load_processors(
            checkpoint, policy_config, self.device,
        )
        self.policy = load_pi05_policy(checkpoint, policy_config, pretrained_base=config.get("pretrained_base", False))
        self.policy.eval().requires_grad_(False)
        self.image_resolution = tuple(policy_config.image_resolution)
        self.horizon = policy_config.chunk_size
        self.dimension = policy_config.action_feature.shape[0]
        self._metadata.update(
            frozen=True, weights_sha256=digest, image_mode="RGB", image_encodings=["png_base64", "png_bytes"],
            feature_kind="normalized_proprioception", context_cache_kind="processed_observation",
            processor_sha256=processor_digest, seed_scope="per_observation", batching=dict(native_model_batching=True),
            model_config_sha256=fingerprint(dict(config=_read_json(checkpoint / "config.json"), io=contract,
                                                processors=processor_digest, steps=policy_config.num_inference_steps,
                                                pretrained_base=config.get("pretrained_base", False))),
            source_revision=fingerprint(dict(lerobot=lerobot_version, adapter=_sha256(Path(__file__)))),
        )

    def describe(self):
        return deepcopy(self._metadata)

    def validate_observations(self, observations):
        for row in observations:
            validate_observation(row, self._metadata["observation"])
            if row.get("model_inputs"):
                raise ValueError("π0.5 does not accept extra model_inputs")

    def _camera_batch(self, observations, camera):
        import torch
        from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

        frames = []
        for row in observations:
            image = (row.decoded_images[camera] if isinstance(row, ValidatedObservation)
                     else decode_image(row["images"][camera], require_rgb=True))
            frame = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).to(self.device, torch.float32).div_(255.)
            if tuple(frame.shape[:2]) != self.image_resolution:
                frame = resize_with_pad_torch(frame, *self.image_resolution)[0]
            frames.append(frame.permute(2, 0, 1))
        return torch.stack(frames)

    def encode_batch(self, observations, seeds):
        import torch

        if len(observations) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
            raise ValueError("Expected one integer seed in [0, 2**63) per π0.5 observation")
        if not observations:
            return []
        self.validate_observations(observations)
        state = torch.from_numpy(finite_array(
            [row["state"] for row in observations], (len(observations), len(self.state_scale)), "observation.state",
        ) * self.state_scale)
        batch = {STATE_KEY: state, "task": [row["task"] for row in observations]}
        with torch.inference_mode():
            for camera, key in self.camera_keys.items():
                batch[key] = self._camera_batch(observations, camera)
            processed = self.preprocessor(batch)
            features = finite_array(processed[STATE_KEY].float().cpu().numpy(), tuple(state.shape), "features")
            # Select model inputs only. Cache compact, per-observation tensors
            # so one surviving context never retains an entire batch allocation.
            from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
            keys = (*self.camera_keys.values(), OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK)
            return [(Pi05Context({key: processed[key][i].detach().clone() for key in keys}, seed), features[i].copy())
                    for i, seed in enumerate(seeds)]

    def decode_batch(self, contexts, initial_noises):
        import torch

        if len(contexts) != len(initial_noises):
            raise ValueError("Expected one optional initial noise per π0.5 context")
        if not contexts:
            return []
        shape = tuple(self._metadata["noise"]["shape"])
        with torch.inference_mode():
            noises = []
            for ctx, noise in zip(contexts, initial_noises):
                if noise is None:
                    noise = torch.randn(shape, dtype=torch.float32, device=self.device,
                                        generator=torch.Generator(device=self.device).manual_seed(ctx.seed))
                else:
                    noise = torch.from_numpy(finite_array(noise, shape, "initial_noise")).to(self.device)
                noises.append(noise)
            batch = {key: torch.stack([ctx.processed[key] for ctx in contexts]) for key in contexts[0].processed}
            prediction = self.policy.predict_action_chunk(batch, noise=torch.stack(noises))
            actions = self.postprocessor(prediction).float().cpu().numpy() * self.action_scale
        actions = finite_array(actions, (len(contexts), self.horizon, self.dimension), "actions")
        return [row.copy() for row in actions]

    def close(self):
        self.policy = self.preprocessor = self.postprocessor = None


def create_backend(config):
    return Pi05Backend(config)
