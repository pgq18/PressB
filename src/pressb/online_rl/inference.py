"""Frozen VLA-JEPA inference addon, without modifying the training repository.

Only this module needs the H200 model environment. Encoder contexts remain on
the inference device; the learner receives pooled features and sends explicit
flow initial noise. The original checkpoint loader retains its integrity checks.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import base64
import hashlib
import importlib
from io import BytesIO
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

import numpy as np
from PIL import Image
import torch


@dataclass(frozen=True)
class EncodedContext:
    tokens: torch.Tensor
    state: torch.Tensor
    feature: np.ndarray
    state_pose9: np.ndarray


@torch.no_grad()
def flow_from_context(head, tokens, state, initial_noise=None):
    """The upstream Euler sampler, with optional explicit initial noise.

The default branch preserves the original randn call and dtype. Explicit noise
replaces that draw; it is never added to another Gaussian sample. Keep operation
order aligned with FlowmatchingActionHead.predict_action.
    """
    tokens = tokens.to(dtype=head.dtype)
    batch_size = tokens.shape[0]
    device = tokens.device
    shape = (batch_size, head.action_horizon, head.action_dim)
    if initial_noise is None:
        actions = torch.randn(size=shape, dtype=tokens.dtype, device=device)
    else:
        actions = torch.as_tensor(initial_noise, dtype=tokens.dtype, device=device)
        if tuple(actions.shape) != shape or not torch.isfinite(actions).all():
            raise ValueError(f"initial_noise must be finite with shape {shape}")
        actions = actions.clone()
    num_steps = head.num_inference_timesteps
    if num_steps is None or num_steps < 1:
        raise ValueError("num_inference_timesteps must be positive")
    dt = 1.0 / num_steps
    state_features = head._encode_state(state, batch_size)
    for step in range(num_steps):
        timestep = int((step / float(num_steps)) * head.num_timestep_buckets)
        timesteps = torch.full(size=(batch_size,), fill_value=timestep, device=device)
        action_features = head.action_encoder(actions, timesteps)
        if head.config.add_pos_embed:
            position_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + head.position_embedding(position_ids).unsqueeze(0)
        future_tokens = head.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        inputs = torch.cat((state_features, future_tokens, action_features), dim=1) \
            if state_features is not None else torch.cat((future_tokens, action_features), dim=1)
        output = head.model(hidden_states=inputs, encoder_hidden_states=tokens, timestep=timesteps)
        prediction = head.action_decoder(output)
        velocity = prediction[:, -head.action_horizon:]
        actions = actions + dt * velocity
    if not torch.isfinite(actions).all():
        raise RuntimeError("The frozen flow model returned nonfinite actions")
    return actions


def _source_identity(repo):
    files = (
        "starVLA/inference/piper_policy.py",
        "starVLA/model/framework/VLA_JEPA.py",
        "starVLA/model/modules/action_model/GR00T_ActionHeader.py",
        "starVLA/model/modules/vlm/QWen3.py",
    )
    identities = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in files}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return dict(vla_repo=str(repo), git_commit=commit, source_sha256=identities,
                addon_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


class FrozenPiperBackend:
    """Separate encoding and flow sampling over a strictly loaded PiperPolicy."""

    def __init__(self, policy, *, state_preparer, pose_converter=None, source_identity=None,
                 image_preprocess_device="cpu"):
        self.policy = policy
        self.model = policy.model
        self.model.requires_grad_(False).eval()
        self.device = torch.device(policy.device)
        if image_preprocess_device not in ("cpu", "cuda"):
            raise ValueError("image_preprocess_device must be cpu or cuda")
        self.image_preprocess_device = image_preprocess_device
        if image_preprocess_device == "cuda":
            if self.device.type != "cuda":
                raise ValueError("CUDA image preprocessing requires a CUDA model device")
            processor = self.model.qwen_vl_interface.processor.image_processor
            if not hasattr(processor, "device") or not hasattr(processor, "disable_grouping"):
                raise ValueError("CUDA image preprocessing requires the saved fast image processor")
            # Runtime execution placement only; keep the checkpoint's resize,
            # normalization, patch layout and tokenizer settings unchanged.
            processor.device = str(self.device)
            processor.disable_grouping = False
        self.state_preparer = state_preparer
        if pose_converter is None:
            from .protocol import pose9_to_pose8
            pose_converter = pose9_to_pose8
        self.pose_converter = pose_converter
        self.source_identity = source_identity or {}
        self.feature_dim = int(self.model.qwen_vl_interface.model.config.hidden_size)
        self.token_count = int(self.model.config.framework.vj2_model.num_embodied_action_tokens_per_instruction)
        self.action_horizon = int(self.model.action_model.action_horizon)
        self.action_dim = int(self.model.action_model.action_dim)
        if self.action_horizon != 7 or self.action_dim != 9:
            raise ValueError("Online PiPER inference requires the trained 7 x 9 action contract")
        if int(self.model.action_model.num_inference_timesteps) != 4:
            raise ValueError("Online PiPER inference requires the original four flow steps")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("The inference model must be completely frozen")

    @classmethod
    def load(cls, vla_repo, checkpoint, *, device="cuda:0", image_preprocess_device="cpu",
             base_vlm=None, base_encoder=None):
        repo = Path(vla_repo).expanduser().resolve(strict=True)
        if not (repo / "starVLA/inference/piper_policy.py").is_file():
            raise ValueError("vla_repo must contain starVLA/inference/piper_policy.py")
        sys.path.insert(0, str(repo))
        module = importlib.import_module("starVLA.inference.piper_policy")
        if not Path(module.__file__).resolve().is_relative_to(repo):
            raise RuntimeError("A different starVLA repository is already imported")
        policy = module.PiperPolicy(checkpoint, device=device, base_vlm=base_vlm, base_encoder=base_encoder)
        backend = cls(policy, state_preparer=module.prepare_state, pose_converter=module.pose9_to_pose8,
                      source_identity=_source_identity(repo), image_preprocess_device=image_preprocess_device)
        if backend.feature_dim != 2048 or backend.token_count != 32:
            raise ValueError("Expected the trained Qwen 32 x 2048 embodied-token contract")
        return backend

    def health(self):
        provenance = self.policy.provenance
        checkpoint = Path(provenance["checkpoint"])
        return dict(checkpoint_verified=True, checkpoint_step=int(checkpoint.name.removeprefix("step_")),
                    checkpoint_sha256=provenance["model_sha256"],
                    model_sha256=provenance["model_sha256"], provenance=provenance,
                    source_identity=self.source_identity, device=str(self.device),
                    frozen=all(not p.requires_grad for p in self.model.parameters()),
                    feature_dim=self.feature_dim, feature_schema="mean_embodied_action_tokens",
                    conditioning_shape=[self.token_count, self.feature_dim],
                    action_horizon=7, action_dim=9, state_dim=9, fps=30,
                    num_inference_timesteps=4, pose_frame="base_link", pose_link="gripper_tcp",
                    xyz_units="metres", rotation6d="first two matrix rows, row-major",
                    quaternion_order="wxyz", normalization="none", gripper_is_learned=False,
                    controller_gripper_width_m=float(self.policy.controller_gripper_width_m),
                    camera_order=["global", "wrist"], resolution=int(self.policy.resolution),
                    image_preprocess_device=self.image_preprocess_device,
                    seed_scope="per_sample", encoder_batch_mode="serial")

    @torch.inference_mode()
    def encode(self, observation, *, seed, include_base_actions=False):
        """Use the old per-sample seed scope, including the Qwen pass."""
        self.model.eval()
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] \
            if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            measured = self.state_preparer(observation["state"])
            images = [image.convert("RGB").resize(
                (self.policy.resolution, self.policy.resolution), Image.Resampling.BILINEAR,
            ) for image in (observation["global_image"], observation["wrist_image"])]
            batch_images = [images]
            image_size = getattr(self.model.config.datasets.vla_data, "image_size", None)
            if image_size:
                from starVLA.training.trainer_utils.trainer_tools import resize_images
                batch_images = resize_images(batch_images, target_size=image_size)
            qwen_inputs = self.model.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, instructions=[observation["task"]],
                prompt_replace_dict={"{actions}": self.model.replace_prompt,
                                     "{e_actions}": self.model.embodied_replace_prompt},
            )
            ids = qwen_inputs["input_ids"]
            mask = torch.isin(ids, torch.tensor([self.model.embodied_action_token_id], device=ids.device))
            if ids.shape[0] != 1 or int(mask.sum()) != self.token_count:
                raise ValueError("Processor changed the trained embodied action token count")
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                output = self.model.qwen_vl_interface(
                    **qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True,
                )
                hidden = output.hidden_states[-1]
                tokens = hidden[mask].reshape(1, self.token_count, self.feature_dim)
            tokens = tokens.to(dtype=self.model.action_model.dtype).detach().clone()
            state = torch.as_tensor(np.array(measured), device=tokens.device,
                                    dtype=self.model.action_model.dtype).clone()
            if not torch.isfinite(tokens).all() or state.shape != (1, 1, 9):
                raise RuntimeError("Invalid frozen encoder context")
            context = EncodedContext(tokens=tokens, state=state,
                                     feature=tokens.float().mean(dim=1)[0].cpu().numpy().copy(),
                                     state_pose9=state.float()[0, 0].cpu().numpy().copy())
            base = self.decode(context) if include_base_actions else None
        return context, base

    @torch.inference_mode()
    def decode(self, context, initial_noise=None):
        self.model.eval()
        noise = None if initial_noise is None else np.asarray(initial_noise, dtype=np.float32)[None]
        with torch.autocast(self.device.type, enabled=False):
            actions = flow_from_context(self.model.action_model, context.tokens, context.state, noise)
        return actions.float()[0].cpu().numpy().copy()

    @torch.inference_mode()
    def encode_batch(self, observations, *, seeds, include_base_actions=False):
        """One multimodal forward for the whole batch, with per-sample flow RNG.

        The frozen encoder is deterministic in eval mode. Each flow draw uses
        its own generator and the original [1,7,9] shape, so changing request
        grouping never changes a sample's initial noise. Floating-point kernels
        can nevertheless differ between batch sizes; serial remains available
        for byte-identical historical evaluation.
        """
        if not observations or len(observations) != len(seeds):
            raise ValueError("Expected one seed for every observation")
        self.model.eval()
        measured = np.concatenate([self.state_preparer(item["state"]) for item in observations], axis=0)
        batch_images = [[image.convert("RGB").resize(
            (self.policy.resolution, self.policy.resolution), Image.Resampling.BILINEAR,
        ) for image in (item["global_image"], item["wrist_image"])] for item in observations]
        image_size = getattr(self.model.config.datasets.vla_data, "image_size", None)
        if image_size:
            from starVLA.training.trainer_utils.trainer_tools import resize_images
            batch_images = resize_images(batch_images, target_size=image_size)
        inputs = self.model.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=[item["task"] for item in observations],
            prompt_replace_dict={"{actions}": self.model.replace_prompt,
                                 "{e_actions}": self.model.embodied_replace_prompt},
        )
        ids = inputs["input_ids"]
        mask = ids == self.model.embodied_action_token_id
        batch_size = len(observations)
        if ids.shape[0] != batch_size or not torch.all(mask.sum(dim=1) == self.token_count):
            raise ValueError("Processor changed the trained embodied action token count")
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            output = self.model.qwen_vl_interface(
                **inputs, output_attentions=False, output_hidden_states=True, return_dict=True,
            )
            tokens = output.hidden_states[-1][mask].reshape(batch_size, self.token_count, self.feature_dim)
        tokens = tokens.to(dtype=self.model.action_model.dtype)
        state = torch.as_tensor(measured, device=tokens.device, dtype=self.model.action_model.dtype)
        if not torch.isfinite(tokens).all() or state.shape != (batch_size, 1, 9):
            raise RuntimeError("Invalid frozen encoder context")
        features = tokens.float().mean(dim=1).cpu().numpy()
        states = state.float()[:, 0].cpu().numpy()
        contexts = [EncodedContext(tokens=tokens[index:index + 1].clone(),
                                   state=state[index:index + 1].clone(),
                                   feature=features[index].copy(), state_pose9=states[index].copy())
                    for index in range(batch_size)]
        bases = [None] * batch_size
        if include_base_actions:
            noise = torch.cat([torch.randn(
                (1, self.action_horizon, self.action_dim), device=tokens.device, dtype=tokens.dtype,
                generator=torch.Generator(device=tokens.device).manual_seed(seed),
            ) for seed in seeds], dim=0)
            with torch.autocast(self.device.type, enabled=False):
                bases = flow_from_context(self.model.action_model, tokens, state, noise).float().cpu().numpy()
        return [(context, base) for context, base in zip(contexts, bases)]

    @torch.inference_mode()
    def decode_batch(self, contexts, initial_noises):
        """Run all four Euler steps once per batch over full cached contexts."""
        if not contexts or len(contexts) != len(initial_noises):
            raise ValueError("Expected one initial-noise array for every context")
        self.model.eval()
        tokens = torch.cat([context.tokens for context in contexts], dim=0)
        state = torch.cat([context.state for context in contexts], dim=0)
        noise = np.stack(initial_noises).astype(np.float32, copy=False)
        with torch.autocast(self.device.type, enabled=False):
            actions = flow_from_context(self.model.action_model, tokens, state, noise)
        return actions.float().cpu().numpy().copy()


class InferenceApplication:
    """Single-threaded HTTP handlers and a bounded, expiring GPU-context cache."""

    def __init__(self, backend, *, cache_size=256, cache_ttl=600.0, batch_size=1, clock=time.monotonic):
        if type(cache_size) is not int or cache_size < 1:
            raise ValueError("cache_size must be a positive integer")
        if not math.isfinite(cache_ttl) or cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive and finite")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.backend = backend
        self.cache_size = cache_size
        self.cache_ttl = float(cache_ttl)
        self.batch_size = batch_size
        self.clock = clock
        self.contexts = OrderedDict()
        self.encoded_count = self.decoded_count = 0

    def _purge(self):
        now = self.clock()
        for key in [key for key, (expires, _) in self.contexts.items() if expires <= now]:
            del self.contexts[key]

    @staticmethod
    def _fields(payload, allowed, required=None):
        if not isinstance(payload, dict) or set(payload) - (set(allowed) | {"protocol_version"}):
            raise ValueError("Unexpected request fields; expected a JSON object")
        if "protocol_version" in payload and (type(payload["protocol_version"]) is not int or payload["protocol_version"] != 1):
            raise ValueError("Incompatible protocol_version")
        if not set(allowed if required is None else required).issubset(payload):
            raise ValueError("Missing request fields")

    def health(self, payload=None):
        self._purge()
        backend_health = dict(self.backend.health())
        backend_health.update(encoder_batch_mode="batched" if self.batch_size > 1 else "serial",
                              microbatch_size=self.batch_size,
                              torch_threads=torch.get_num_threads(),
                              feature_encodings=["json", "float32_base64"],
                              accepted_image_sizes=[[640, 480], [224, 224]],
                              encode_release_context_ids=True)
        return dict(service="inference", ready=True, **backend_health,
                    cache=dict(size=len(self.contexts), capacity=self.cache_size, ttl_seconds=self.cache_ttl,
                               eviction="oldest_created", ttl_refresh_on_decode=False),
                    encoded_count=self.encoded_count, decoded_count=self.decoded_count)

    def _observation(self, observation):
        self._fields(observation, ("task", "state", "images", "control_state"), ("task", "state", "images"))
        if not isinstance(observation["task"], str) or not re.fullmatch(r"Press (?:2[4-9]|3[0-5]) floor\.", observation["task"]):
            raise ValueError("Expected an exact trained task for floors 24..35")
        state = observation["state"]
        if (not isinstance(state, list) or len(state) != 8
                or any(type(value) not in (int, float) for value in state)):
            raise ValueError("observation.state must be a numeric raw8 vector")
        state = np.asarray(state, dtype=np.float32)
        if not np.isfinite(state).all():
            raise ValueError("observation.state must be finite")
        self.backend.state_preparer(state)
        images = observation["images"]
        if not isinstance(images, dict) or set(images) != {"global", "wrist"}:
            raise ValueError("images must contain exactly global and wrist")
        decoded = {}
        for key, value in images.items():
            if not isinstance(value, str) or len(value) > 12_000_000:
                raise ValueError("Expected a bounded base64 PNG image")
            with Image.open(BytesIO(base64.b64decode(value, validate=True))) as picture:
                if picture.format != "PNG" or picture.size not in ((640, 480), (224, 224)) or picture.mode != "RGB":
                    raise ValueError("Expected a lossless RGB PNG of size 640x480 or 224x224")
                decoded[key] = picture.copy()
        for key, value in decoded.items():
            if isinstance(value, np.ndarray):
                value = Image.fromarray(value)
            if not isinstance(value, Image.Image):
                raise ValueError("decode_image must return RGB image data")
            decoded[key] = value
        return dict(task=observation["task"], state=state,
                    global_image=decoded["global"], wrist_image=decoded["wrist"])

    def _actions(self, actions):
        if actions.shape != (7, 9) or not np.isfinite(actions).all():
            raise RuntimeError("Frozen policy must return finite [7,9] actions")
        pose8 = self.backend.pose_converter(actions)
        return actions.tolist(), np.asarray(pose8).tolist()

    def encode(self, payload):
        self._fields(payload, ("observations", "seeds", "include_base_actions", "release_context_ids", "feature_encoding"),
                     ("observations", "seeds", "include_base_actions"))
        observations, seeds = payload["observations"], payload["seeds"]
        include = payload["include_base_actions"]
        if (not isinstance(observations, list) or not observations or len(observations) > self.cache_size
                or not isinstance(seeds, list) or len(seeds) != len(observations)
                or type(include) is not bool):
            raise ValueError("Expected 1..cache_size observations, one seed per sample and a boolean base-action flag")
        if any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
            raise ValueError("seeds must contain integers in [0, 2**63)")
        release_keys = payload.get("release_context_ids", [])
        if not isinstance(release_keys, list) or any(not isinstance(key, str) for key in release_keys):
            raise ValueError("release_context_ids must be a list of strings")
        feature_encoding = payload.get("feature_encoding", "json")
        if feature_encoding not in ("json", "float32_base64"):
            raise ValueError("feature_encoding must be json or float32_base64")
        decoded = [self._observation(observation) for observation in observations]
        pending = []
        # Finish the whole request before committing contexts, including pose conversion.
        encoded = []
        for offset in range(0, len(decoded), self.batch_size):
            if self.batch_size == 1:
                encoded.append(self.backend.encode(decoded[offset], seed=seeds[offset], include_base_actions=include))
            else:
                batch = self.backend.encode_batch(decoded[offset:offset + self.batch_size],
                    seeds=seeds[offset:offset + self.batch_size], include_base_actions=include)
                if len(batch) != len(decoded[offset:offset + self.batch_size]):
                    raise RuntimeError("Encoder returned an incorrect batch size")
                encoded.extend(batch)
        for context, base in encoded:
            key = uuid.uuid4().hex
            item = dict(context_id=key, state_pose9=context.state_pose9.tolist())
            if feature_encoding == "float32_base64":
                feature = np.ascontiguousarray(context.feature, dtype="<f4")
                if feature.shape != (self.backend.feature_dim,) or not np.isfinite(feature).all():
                    raise RuntimeError("Invalid frozen encoder feature")
                item.update(feature_f32_b64=base64.b64encode(feature.tobytes()).decode("ascii"),
                            feature_dim=len(feature))
            else:
                item["feature"] = context.feature.tolist()
            if include:
                item["base_actions_pose9"], item["base_actions_pose8"] = self._actions(base)
            pending.append((key, context, item))
        self._purge()
        for key in dict.fromkeys(release_keys):
            self.contexts.pop(key, None)
        while len(self.contexts) + len(pending) > self.cache_size:
            self.contexts.popitem(last=False)
        for key, context, _ in pending:
            self.contexts[key] = (self.clock() + self.cache_ttl, context)
        self.encoded_count += len(pending)
        return dict(items=[item for _, _, item in pending])

    def decode(self, payload):
        self._fields(payload, ("items",))
        items = payload["items"]
        if not isinstance(items, list) or not items or len(items) > self.cache_size:
            raise ValueError("items must contain 1..cache_size contexts")
        self._purge()
        requests = []
        for item in items:
            self._fields(item, ("context_id", "initial_noise"))
            key = item["context_id"]
            if not isinstance(key, str) or key not in self.contexts:
                raise ValueError("Context is missing or expired; encode the current observation explicitly")
            raw_noise = item["initial_noise"]
            if (not isinstance(raw_noise, list) or len(raw_noise) != 7
                    or any(not isinstance(row, list) or len(row) != 9
                           or any(type(value) not in (int, float) for value in row) for row in raw_noise)):
                raise ValueError("initial_noise must be a numeric [7,9] array")
            noise = np.asarray(raw_noise, dtype=np.float32)
            if not np.isfinite(noise).all():
                raise ValueError("initial_noise must be finite")
            requests.append((key, self.contexts[key][1], noise))
        decoded = []
        for offset in range(0, len(requests), self.batch_size):
            batch = requests[offset:offset + self.batch_size]
            if self.batch_size == 1:
                decoded.append(self.backend.decode(batch[0][1], batch[0][2]))
            else:
                actions = self.backend.decode_batch([item[1] for item in batch], [item[2] for item in batch])
                if len(actions) != len(batch):
                    raise RuntimeError("Decoder returned an incorrect batch size")
                decoded.extend(actions)
        result = []
        for (key, _, _), actions in zip(requests, decoded):
            actions9, actions8 = self._actions(actions)
            result.append(dict(context_id=key, actions_pose9=actions9, actions_pose8=actions8))
        self.decoded_count += len(result)
        return dict(items=result)

    def release(self, payload):
        self._fields(payload, ("context_ids",))
        keys = payload["context_ids"]
        if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys):
            raise ValueError("context_ids must be a list of strings")
        self._purge()
        released = sum(self.contexts.pop(key, None) is not None for key in dict.fromkeys(keys))
        return dict(released=released)

    @property
    def handlers(self):
        return {"/health": self.health, "/encode": self.encode, "/decode": self.decode, "/release": self.release}
