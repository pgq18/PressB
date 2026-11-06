"""Cartesian VLA-JEPA adapter with explicit instruction/pose contracts.

The legacy factory remains elevator-specific and uses its strict checkpoint
loader. Joint checkpoints use ``vlajepa_joint``; they cannot enter this adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np

from pressb.runtime.contract import decode_image, finite_array, fingerprint, validate_observation
from pressb.runtime.image_transport import ValidatedObservation


ELEVATOR_INSTRUCTION_PATTERN = r"Press (?:2[4-9]|3[0-5]) floor\."


@dataclass(frozen=True)
class _Context:
    encoded: object
    baseline_noise: np.ndarray


class VLAJEPABackend:
    def __init__(self, frozen_backend, *, contract_profile="elevator_pose9", instruction_pattern=None):
        if contract_profile not in ("elevator_pose9", "pose9"):
            raise ValueError("VLA-JEPA Cartesian contract_profile must be elevator_pose9 or pose9")
        if frozen_backend.action_dim != 9 or frozen_backend.action_horizon < 1:
            raise ValueError("The pose9 adapter requires a nine-dimensional Cartesian head; use vlajepa_joint for joints")
        if contract_profile == "elevator_pose9":
            if instruction_pattern not in (None, ELEVATOR_INSTRUCTION_PATTERN):
                raise ValueError("The elevator_pose9 profile retains its trained instruction pattern")
            instruction_pattern = ELEVATOR_INSTRUCTION_PATTERN
        if instruction_pattern is not None and (not isinstance(instruction_pattern, str) or not instruction_pattern):
            raise ValueError("instruction_pattern must be a nonempty regular expression or null")
        self.instruction_pattern = None if instruction_pattern is None else re.compile(instruction_pattern)
        self.contract_profile = contract_profile
        self.frozen = frozen_backend
        source = frozen_backend.health()
        self._metadata = dict(
            schema_version=2, node_type="inference",
            model_id="vlajepa:" + source["checkpoint_sha256"],
            capabilities=["encode", "decode", "infer", "external_initial_noise", "features", "context_cache"],
            observation=dict(
                state=dict(names=["x", "y", "z", "qw", "qx", "qy", "qz", "gripper_width"],
                           units=["m"] * 3 + ["unitless"] * 4 + ["m"]),
                cameras=["global", "wrist"]),
            action=dict(representation="pose9", frame="robot_base",
                        names=["x", "y", "z"] + [f"rot6d_{i}" for i in range(6)],
                        units=["m"] * 3 + ["unitless"] * 6,
                        horizon=frozen_backend.action_horizon, control_hz=30),
            noise=dict(shape=[frozen_backend.action_horizon, frozen_backend.action_dim],
                       semantics="flow_initial_state", baseline="per_observation_seeded_gaussian"),
            feature_dim=frozen_backend.feature_dim,
            feature_schema="mean_embodied_action_tokens", frozen=source["frozen"],
            checkpoint_sha256=source["checkpoint_sha256"],
            checkpoint_step=source["checkpoint_step"], provenance=source.get("provenance", {}),
            source_identity=source.get("source_identity", {}),
            source_revision=fingerprint({
                "model_source": {k: v for k, v in source.get("source_identity", {}).items() if k != "vla_repo"},
                "adapter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}),
            model_config_sha256=fingerprint({**{k: source.get(k) for k in
                ("num_inference_timesteps", "normalization", "resolution", "image_preprocess_device",
                 "pose_frame", "pose_link", "controller_gripper_width_m")},
                 "contract_profile": contract_profile, "instruction_pattern": instruction_pattern}),
            contract_profile=contract_profile, instruction_pattern=instruction_pattern,
            fixed_gripper_width_m=source["controller_gripper_width_m"],
            rotation6d="first_two_matrix_rows_row_major", quaternion_order="wxyz",
            seed_scope="per_observation", context_cache_kind="embodied_action_tokens_and_proprioception",
        )

    def describe(self):
        from copy import deepcopy
        return deepcopy(self._metadata)

    def validate_observations(self, observations):
        for observation in observations:
            validate_observation(observation, self._metadata["observation"])
            if self.instruction_pattern is not None and not self.instruction_pattern.fullmatch(observation["task"]):
                raise ValueError(f"The {self.contract_profile} checkpoint instruction does not match its configured pattern")
            if observation.get("model_inputs"):
                raise ValueError("This VLA-JEPA checkpoint does not accept extra model_inputs")
            self.frozen.state_preparer(finite_array(observation["state"], (8,), name="state"))

    def encode_batch(self, observations, seeds):
        import torch

        if not observations or len(observations) != len(seeds) or any(
                type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
            raise ValueError("Expected one integer seed in [0, 2**63) per observation")
        self.validate_observations(observations)
        prepared = [dict(task=observation["task"],
                         state=finite_array(observation["state"], (8,), name="state"),
                         global_image=(observation.decoded_images["global"] if isinstance(observation, ValidatedObservation)
                                       else decode_image(observation["images"]["global"])),
                         wrist_image=(observation.decoded_images["wrist"] if isinstance(observation, ValidatedObservation)
                                      else decode_image(observation["images"]["wrist"])))
                    for observation in observations]
        encoded = self.frozen.encode_batch(prepared, seeds=seeds, include_base_actions=False)
        if len(encoded) != len(observations):
            raise ValueError("VLA-JEPA returned an incorrect encoder batch size")
        results = []
        for (context, _), seed in zip(encoded, seeds):
            # Preserve the original sampler's [1,H,D] draw shape and model dtype.
            noise = torch.randn(
                (1, self.frozen.action_horizon, self.frozen.action_dim),
                device=context.tokens.device, dtype=context.tokens.dtype,
                generator=torch.Generator(device=context.tokens.device).manual_seed(seed),
            ).float()[0].cpu().numpy().copy()
            noise.setflags(write=False)
            feature = finite_array(context.feature, (self.frozen.feature_dim,), name="features")
            results.append((_Context(context, noise), feature.copy()))
        return results

    def decode_batch(self, contexts, initial_noises):
        if not contexts or len(contexts) != len(initial_noises):
            raise ValueError("Expected one noise entry per context")
        shape = tuple(self._metadata["noise"]["shape"])
        noises = [context.baseline_noise if noise is None else
                  finite_array(noise, shape, name="initial_noise")
                  for context, noise in zip(contexts, initial_noises)]
        actions = finite_array(self.frozen.decode_batch([context.encoded for context in contexts], noises),
                               (len(contexts), self.frozen.action_horizon, 9), name="actions")
        return [row.copy() for row in actions]

    def close(self):
        close = getattr(self.frozen, "close", None)
        if callable(close):
            close()


def create_backend(config):
    """Load a frozen elevator checkpoint from explicit configuration paths."""
    from .vlajepa_backend import FrozenPiperBackend

    allowed = {"vla_repo", "checkpoint", "device", "image_preprocess_device", "base_vlm", "base_encoder"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError(f"VLA-JEPA backend configuration supports {sorted(allowed)}")
    if "checkpoint" not in config:
        raise ValueError("VLA-JEPA backend requires checkpoint")
    return VLAJEPABackend(FrozenPiperBackend.load(**config))
