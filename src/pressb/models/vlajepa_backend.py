"""Frozen VLA-JEPA model encoding and flow sampling.

Encoder contexts remain on the inference device. The checkpoint loader retains
its integrity checks and supports both bundled and explicit model sources.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import hashlib
import importlib
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch

from pressb.model_sources import activate_source, builtin_source_root, source_tree_sha256


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
    bundled = repo.resolve() == builtin_source_root("vlajepa").resolve()
    commit = None
    if not bundled:
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    identity = dict(vla_repo=str(repo), git_commit=commit, source_sha256=identities,
                    addon_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if bundled:
        # A bundled tree belongs to PressB's Git checkout. Its parent HEAD is
        # not an upstream VLA-JEPA revision and must never be reported as one.
        identity.update(source_distribution="pressb_builtin",
                        vendored_manifest_sha256=hashlib.sha256((repo / "source_manifest.json").read_bytes()).hexdigest(),
                        bundled_model_source_sha256=source_tree_sha256("vlajepa", repo))
    return identity


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
            from pressb.control.pose_protocol import pose9_to_pose8
            # Pose9 carries no gripper command; the trained policy owns it.
            pose_converter = partial(pose9_to_pose8, gripper_width_m=policy.controller_gripper_width_m)
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
    def load(cls, vla_repo=None, checkpoint=None, *, device="cuda:0", image_preprocess_device="cpu",
             base_vlm=None, base_encoder=None):
        if checkpoint is None:
            raise ValueError("checkpoint is required")
        repo = activate_source("vlajepa", vla_repo)
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

