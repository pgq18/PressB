"""Schema-v2 adapter for the bundled visual Diffusion Policy."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from pressb.runtime.contract import decode_image, finite_array, validate_observation, fingerprint
from pressb.runtime.image_transport import ValidatedObservation


@dataclass
class DiffusionContext:
    features: object
    initial_noise: object
    seed: int


class DiffusionPolicyBackend:
    def __init__(self, config):
        import torch
        from pressb.model_sources import source_tree_sha256
        from pressb.training.diffusion_policy import load_checkpoint
        from pressb.training.diffusion_policy_data import file_hash

        allowed = {"checkpoint", "device", "use_ema", "kernel", "denoising_steps", "eta", "min_std", "model_id"}
        if not isinstance(config, dict) or set(config) - allowed:
            raise ValueError(f"Diffusion Policy supports configuration keys {sorted(allowed)}")
        if type(config.get("use_ema", True)) is not bool:
            raise ValueError("use_ema must be boolean")
        variant = "ema" if config.get("use_ema", True) else "model"
        self.device = torch.device(config.get("device", "cuda:0"))
        self.model, checkpoint = load_checkpoint(config["checkpoint"], self.device, use_ema=config.get("use_ema", True))
        self.model.eval().requires_grad_(False)
        self.config = self.model.config
        self.horizon = self.model.horizon
        self.dimension = self.model.action_dim
        self.feature_dim = self.model.feature_dim
        self.sampler_config = {"kernel": config.get("kernel", "ddim"),
                               "denoising_steps": config.get("denoising_steps", self.config.get("inference_steps", 10)),
                               "eta": config.get("eta", 0.), "min_std": config.get("min_std", 0.)}
        self.expert = self.model.action_expert(self.sampler_config).eval()
        digest = file_hash(config["checkpoint"])
        self._metadata = dict(schema_version=2, node_type="inference",
            model_id=config.get("model_id", "diffusion_policy:" + digest + ":" + variant),
            capabilities=["encode", "decode", "infer", "external_initial_noise", "features", "context_cache", "trainable_action_expert"],
            observation=dict(state=dict(names=self.config["state_names"], units=self.config["state_units"]),
                             cameras=self.config["cameras"]),
            action=dict(representation="joint_position", frame="robot_base", names=self.config["action_names"],
                        units=self.config["action_units"], horizon=self.horizon, control_hz=self.config.get("control_hz", 30)),
            noise=dict(shape=[self.horizon, self.dimension], semantics="diffusion_initial_state"),
            feature_dim=self.feature_dim, feature_kind="resnet18_spatial_softmax_state_task_embedding",
            frozen=True, weights_sha256=digest, checkpoint_step=checkpoint["step"], checkpoint_variant=variant,
            normalization_sha256=fingerprint(checkpoint["normalization"]),
            model_config_sha256=fingerprint(dict(model=self.config, sampler=self.sampler_config, checkpoint_variant=variant)),
            source_revision=fingerprint(dict(source=source_tree_sha256("diffusion_policy"),
                                           adapter=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())),
            image_mode="RGB", image_encodings=["png_base64", "png_bytes"],
            seed_scope="per_observation", batching=dict(native_model_batching=True),
            sampler=self.expert.spec(), training_dataset_sha256=checkpoint["training_contract"]["dataset_sha256"],
            task_instructions=self.config["tasks"])

    def describe(self):
        from copy import deepcopy
        return deepcopy(self._metadata)

    def validate_observations(self, observations):
        for row in observations:
            validate_observation(row, self._metadata["observation"])
            if row["task"] not in self.config["tasks"]:
                raise ValueError("Instruction is outside this Diffusion Policy checkpoint's trained task vocabulary")
            if row.get("model_inputs"):
                raise ValueError("Diffusion Policy does not accept extra model_inputs")

    def encode_batch(self, observations, seeds):
        import torch
        from PIL import Image
        if len(observations) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
            raise ValueError("Each observation requires an integer seed in [0, 2**63)")
        if not observations:
            return []
        self.validate_observations(observations)
        images = []
        for row in observations:
            camera_images = []
            for camera in self.config["cameras"]:
                image = (row.decoded_images[camera] if isinstance(row, ValidatedObservation)
                         else decode_image(row["images"][camera]))
                image = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize(
                    (self.config["resolution"], self.config["resolution"]), Image.Resampling.BILINEAR)
                camera_images.append(np.asarray(image).transpose(2, 0, 1).copy())
            images.append(np.stack(camera_images))
        states = torch.as_tensor(np.asarray([row["state"] for row in observations]), device=self.device, dtype=torch.float32)
        task_ids = torch.tensor([self.config["tasks"].index(row["task"]) for row in observations], device=self.device)
        with torch.inference_mode():
            encoded = self.model.encode(torch.as_tensor(np.stack(images), device=self.device), states, task_ids)
        result = []
        for feature, seed in zip(encoded, seeds):
            noise = torch.randn((self.horizon, self.dimension), device=self.device,
                                generator=torch.Generator(device=self.device).manual_seed(seed))
            result.append((DiffusionContext(feature.detach().clone(), noise, seed), feature.float().cpu().numpy().copy()))
        return result

    def pack_contexts(self, contexts):
        import torch
        return {"features": torch.stack([context.features for context in contexts])}

    def create_action_expert(self, config=None):
        # The caller can deep-copy this head for a separate optimizer. Encoder
        # parameters are never registered in the action-expert module.
        self.model.encoder.requires_grad_(False)
        expert = self.model.action_expert(config or {}).eval()
        expert.network.requires_grad_(True)
        return expert

    def decode_batch(self, contexts, initial_noises):
        import torch
        if len(contexts) != len(initial_noises):
            raise ValueError("One optional initial noise is required per context")
        if not contexts:
            return []
        latent = torch.stack([context.initial_noise if noise is None else torch.as_tensor(
            finite_array(noise, (self.horizon, self.dimension), name="initial_noise"), device=self.device)
            for context, noise in zip(contexts, initial_noises)])
        generators = [torch.Generator(device=self.device).manual_seed(context.seed ^ 0x5DEECE66D) for context in contexts]
        with torch.inference_mode():
            condition = self.pack_contexts(contexts)
            for step in range(self.expert.denoising_steps):
                mean, std = self.expert.transition(latent, step, condition)
                if self.expert.eta or self.expert.min_std or self.expert.kernel == "ddpm":
                    noise = torch.stack([torch.randn((self.horizon, self.dimension), generator=g, device=self.device) for g in generators])
                    latent = mean + std * noise
                else:
                    latent = mean
            actions = self.expert.actions_from_latent(latent).float().cpu().numpy()
        finite_array(actions, (len(contexts), self.horizon, self.dimension), name="actions")
        return [action.copy() for action in actions]

    def close(self):
        self.model = self.expert = None


def create_backend(config):
    return DiffusionPolicyBackend(config)
