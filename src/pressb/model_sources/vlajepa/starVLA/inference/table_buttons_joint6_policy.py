"""Six learned arm actions with an explicit fixed gripper on the joint7 wire.

The network has state_dim=7 and action_dim=6. Only this transport adapter adds
the constant full-width gripper command; no gripper noise, prediction or
learned output exists. The existing joint7 policy implementation is unchanged.
Noise is drawn as [1,7,6] per sample. The response includes those same [7,6]
float32 values in initial_noise, and initial_noise_sha256 hashes their C-order
float32 bytes (before any integration). A simulator audit must use native six
dimensions when reconstructing the noise, despite the seven-column wire.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import time

import numpy as np

from starVLA.inference.table_buttons_policy import (
    CAMERAS, SCALE as WIRE_SCALE, SCHEMA as WIRE_SCHEMA, InvalidPolicyAction,
    TableButtonsPolicy, flow_with_noise, read_json, seeded_noise, sha256,
)

NATIVE_SCHEMA = "vlajepa.table_buttons.joint6.v1"
MODEL_ID = "medium_red_joint6"
MODEL_GROUPS = {MODEL_ID: "medium_red", **{f"{spacing}_red_joint6_1000": f"{spacing}_red"
                                         for spacing in ("near", "medium", "far")}}
ACTION_SCALE = np.full(6, 180., dtype=np.float32)
# Preserve the source action's float32 centimetres and the executor's * .01
# conversion; rounding 0.0002 directly to float32 metres gives a different bit pattern.
FIXED_GRIPPER_FULL_WIDTH_M = float(np.float32(.02)) * .01


def validate_normalization(normalization):
    """Reject interpreting a six-action checkpoint as the old seven-action head."""
    if normalization.get("kind") != "separate_joint7_state_joint6_action" or normalization.get("version") != 1:
        raise ValueError("Expected separate joint7-state/joint6-action normalization")
    names = [f"joint_{i}.pos" for i in range(1, 7)]
    for key, expected_names, units, scale, kind in (
        ("state", names + ["gripper.pos"], ["degree"] * 6 + ["cm"], WIRE_SCALE.tolist(), "fixed_joint7_scale"),
        ("action", names, ["degree"] * 6, ACTION_SCALE.tolist(), "fixed_joint6_scale"),
    ):
        spec = normalization[key]
        if (spec.get("kind") != kind or spec.get("version") != 1
                or spec.get("feature_names") != expected_names or spec.get("input_units") != units
                or spec.get("scale") != scale or spec.get("offset") != [0.] * len(scale)
                or spec.get("clip") is not False):
            raise ValueError(f"Invalid fixed {key} normalization")
    if normalization["state"].get("gripper_representation") != "full_width":
        raise ValueError("State gripper must be full width")
    width = normalization.get("fixed_gripper_full_width_m")
    if (isinstance(width, bool) or not isinstance(width, (int, float))
            or not np.isfinite(width) or width != FIXED_GRIPPER_FULL_WIDTH_M):
        raise ValueError("Fixed gripper must equal source float32 0.02 centimetres converted to metres")
    return float(width)


def expand_joint6_actions(normalized_arm, fixed_gripper_full_width_m):
    """Native (...,7,6) -> wire (...,7,7), degrees and full-width centimetres.

    The final normalized transport coordinate is fixed_width_m*100/8. Arm
    values are not clipped or smoothed. This is not another learned channel.
    """
    arm = np.asarray(normalized_arm, dtype=np.float32)
    if arm.ndim != 3 or arm.shape[1:] != (7, 6) or not np.isfinite(arm).all():
        raise InvalidPolicyAction("Expected finite native [batch,7,6] actions")
    if fixed_gripper_full_width_m != FIXED_GRIPPER_FULL_WIDTH_M:
        raise ValueError("Fixed gripper command differs from dataset contract")
    width_cm = fixed_gripper_full_width_m * 100.
    normalized = np.concatenate((arm, np.full((*arm.shape[:-1], 1), width_cm / 8., dtype=np.float32)), axis=-1)
    physical = normalized * WIRE_SCALE
    if not np.isfinite(physical).all():
        raise InvalidPolicyAction("Nonfinite physical joint actions after denormalization")
    return normalized, physical


def load_metadata(run, model_id=None, checkpoint=None):
    """Read native metadata; archived or in-run checkpoints are both supported.

    This does not read the full model. A missing published digest is computed
    once during actual model loading and returned to the client before use.
    """
    from omegaconf import OmegaConf
    run = Path(run).resolve(strict=True)
    config = OmegaConf.load(run / "config.yaml")
    schema, normalization = read_json(run / "action_schema.json"), read_json(run / "normalization.json")
    checkpoint = Path(checkpoint).resolve(strict=True) if checkpoint else (run / "checkpoints/best").resolve(strict=True)
    model_path = checkpoint / "model.pt"
    manifest = read_json(checkpoint / "manifest.json")
    model_id = str(config.run_id) if model_id is None else model_id
    if model_id not in MODEL_GROUPS or config.run_id != model_id:
        raise ValueError("Unknown or mismatched independent red-button joint6 run")
    group = MODEL_GROUPS[model_id]
    if schema.get("schema") != NATIVE_SCHEMA or manifest.get("schema") != NATIVE_SCHEMA:
        raise ValueError("Expected native joint6 schema in action and checkpoint manifests")
    for key, value in (("state_dim", 7), ("action_dim", 6), ("action_horizon", 7)):
        if schema.get(key) != value or int(getattr(config.framework.action_model, key)) != value:
            raise ValueError(f"Native model architecture mismatch: {key}")
    if schema.get("cameras") != list(CAMERAS) or schema.get("fps") != 30:
        raise ValueError("Expected up/wrist cameras at 30 Hz")
    width = validate_normalization(normalization)
    if schema.get("normalization") != normalization or schema.get("fixed_gripper_full_width_m") != width:
        raise ValueError("Action schema and fixed normalization differ")
    if not manifest.get("complete") or model_path.stat().st_size != manifest["model_bytes"]:
        raise ValueError("Checkpoint is incomplete or has an unexpected size")
    for key in ("dataset_sha256", "split_sha256", "training_fingerprint"):
        if not schema.get(key) or schema[key] != manifest.get(key):
            raise ValueError(f"Checkpoint provenance mismatch: {key}")
    if schema.get("dataset_group", "medium_red" if model_id == MODEL_ID else None) != group:
        raise ValueError("Joint6 model dataset group differs from its run ID")
    if model_id != MODEL_ID:
        data_root = Path(config.datasets.vla_data.data_root_dir)
        if data_root.name != group:
            raise ValueError("Configuration dataset group differs from the joint6 model")
        # Count provenance remains in the saved training contract; inference
        # does not require the original training dataset to remain mounted.
        contract = read_json(run / "training_contract.json")["contract"]
        if contract.get("source_episode_count") != 1000 or contract.get("split_counts") != {"train": 900, "val": 100}:
            raise ValueError("Expanded joint6 model must retain the 1000-episode 900/100 training contract")
    digest = manifest.get("model_sha256")
    archive_path = run / "archive.json"
    if archive_path.is_file():
        archive = read_json(archive_path)
        # An explicit last/probe checkpoint need not be the archived best model.
        if Path(archive["model_path"]).resolve() == model_path.resolve():
            if archive["model_bytes"] != model_path.stat().st_size or archive["best_step"] != manifest["step"]:
                raise ValueError("Archived model metadata differs from selected checkpoint")
            if digest and digest != archive["model_sha256"]:
                raise ValueError("Archive and checkpoint digests differ")
            digest = archive["model_sha256"]
    identity = dict(model_id=model_id, spacing=group.split("_")[0], task_type="single", source_dataset_group=group,
        source_episode_count=100 if model_id == MODEL_ID else 1000,
        allowed_instructions=["Press the red button."], model_sha256=digest,
        model_bytes=model_path.stat().st_size, best_step=manifest["step"], checkpoint_step=manifest["step"],
        checkpoint_path=str(model_path), run=str(run), camera_order=list(CAMERAS), image_input_size=[640, 480],
        image_resize="PIL RGB BILINEAR square; identical to training adapter",
        image_model_size=[int(config.datasets.vla_data.resolution_size)] * 2,
        state_units=["degree"] * 6 + ["full_width_cm"], action_units=["degree"] * 6 + ["full_width_cm"],
        normalization_scale=WIRE_SCALE.tolist(), state_dim=7, action_dim=7, action_horizon=7,
        wire_schema=WIRE_SCHEMA, wire_action_dim=7, native_schema=NATIVE_SCHEMA, native_action_dim=6,
        noise_shape_per_sample=[1, 7, 6], noise_hash_array_shape=[7, 6], noise_hash_dtype="float32",
        noise_response_field="initial_noise", noise_hash_memory_order="C",
        native_action_units=["degree"] * 6, native_action_normalization_scale=ACTION_SCALE.tolist(),
        action_dim_semantics="action_dim is the transport width; the network has native_action_dim=6",
        fixed_gripper_full_width_m=width, fixed_gripper_full_width_cm=width * 100., gripper_learned=False,
        gripper_rule="append the fixed dataset full-width command after native six-arm-action prediction",
        fps=30, smoothing="none", action_clipping="none", recurrent_state="none",
        num_inference_timesteps=int(config.framework.action_model.num_inference_timesteps))
    return config, identity


class TableButtonsJoint6Policy(TableButtonsPolicy):
    """Reuse joint7 observation preprocessing; load and sample only a joint6 head."""
    metadata_loader = staticmethod(load_metadata)

    def __init__(self, run, model_id=None, device="cuda:0", checkpoint=None):
        import torch
        from transformers import AutoProcessor
        from starVLA.model.framework.VLA_JEPA import VLA_JEPA
        from starVLA.training import train_piper as shared
        from starVLA.inference.piper_policy import install_saved_processor
        started = time.monotonic()
        self.config, self.identity = self.metadata_loader(run, model_id, checkpoint)
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.config.framework.qwenvl.device_map = "cpu"
        if not self.config.framework.qwenvl.init_from_config or not self.config.framework.vj2_model.init_from_config:
            raise ValueError("Expected architecture-only construction before strict load")
        actual_digest = sha256(self.identity["checkpoint_path"])
        if self.identity["model_sha256"] and actual_digest != self.identity["model_sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        self.identity["model_sha256"] = actual_digest
        shared.seed_all(42)
        self.model = VLA_JEPA(config=self.config)
        install_saved_processor(self.model, AutoProcessor.from_pretrained(Path(run) / "processor", local_files_only=True))
        shared.configure_parameters(self.model, checkpointing=False)
        weights = torch.load(self.identity["checkpoint_path"], map_location="cpu", weights_only=True, mmap=True)
        shared.validate_tied_source(self.model, weights)
        if set(weights) != set(self.model.state_dict()):
            raise ValueError("Native joint6 checkpoint keys differ from the model")
        for name, tensor in list(self.model.named_parameters(remove_duplicate=False)) + list(self.model.named_buffers(remove_duplicate=False)):
            if name in weights and tensor.dtype != weights[name].dtype:
                tensor.data = tensor.data.to(weights[name].dtype)
        self.model.load_state_dict(weights, strict=True)
        if any(value.dtype != weights[name].dtype for name, value in self.model.state_dict().items()):
            raise ValueError("Strict loading changed saved tensor dtype")
        self.identity.update(strict_loaded=True, model_digest_verified=True, loaded_keys=len(weights),
                             saved_dtype_counts=dict(Counter(str(value.dtype) for value in weights.values())))
        del weights
        if self.model.action_model.action_dim != 6 or self.model.action_model.config.state_dim != 7:
            raise ValueError("Loaded head does not have action6/state7")
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(self.device)
        self.config.framework.qwenvl.device_map = str(self.device)
        self.resolution = int(self.config.datasets.vla_data.resolution_size)
        self.identity.update(device=str(self.device), loaded_seconds=time.monotonic() - started)

    def predict_prepared(self, images, states, instructions, seeds, *, return_noise=False):
        import torch
        with torch.inference_mode():
            model = self.model
            inputs = model.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions=instructions,
                prompt_replace_dict={"{actions}": model.replace_prompt, "{e_actions}": model.embodied_replace_prompt})
            indices = torch.isin(inputs["input_ids"], torch.tensor([model.embodied_action_token_id], device=inputs["input_ids"].device)).nonzero(as_tuple=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model.qwen_vl_interface(**inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
                hidden = result.hidden_states[-1]
                embeddings = hidden[indices[0], indices[1], :].view(hidden.shape[0], -1, hidden.shape[-1])
            state_tensor = torch.as_tensor(states, device=self.device, dtype=model.action_model.dtype)
            noise = seeded_noise(model.action_model, len(images), seeds, self.device)
            with torch.autocast("cuda", enabled=False):
                actions = flow_with_noise(model.action_model, embeddings, state_tensor, noise)
            native = actions.float().cpu().numpy()
            if native.shape != (len(images), 7, 6) or not np.isfinite(native).all():
                raise InvalidPolicyAction("Nonfinite or incorrectly shaped native joint6 actions")
            initial_noise = noise.float().cpu().numpy()
            noise_hashes = [hashlib.sha256(row.tobytes(order="C")).hexdigest() for row in initial_noise]
            return (native, noise_hashes, initial_noise) if return_noise else (native, noise_hashes)

    def predict(self, samples):
        import torch
        started = time.monotonic()
        images, states, instructions, seeds = self.prepare(samples)
        native, noise_hashes, initial_noise = self.predict_prepared(images, states, instructions, seeds, return_noise=True)
        normalized, physical = expand_joint6_actions(native, self.identity["fixed_gripper_full_width_m"])
        torch.cuda.synchronize(self.device)
        predictions = []
        for index, sample in enumerate(samples):
            predictions.append(dict(episode_id=sample["episode_id"], request_id=sample["request_id"], noise_seed=seeds[index],
                actions=physical[index].tolist(), normalized_actions=normalized[index].tolist(),
                native_normalized_actions=native[index].tolist(), native_actions=(native[index] * ACTION_SCALE).tolist(),
                normalized_state=states[index, 0].tolist(), initial_noise_sha256=noise_hashes[index],
                initial_noise=initial_noise[index].tolist(),
                preprocessed_image_sha256={camera: hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
                                          for camera, image in zip(CAMERAS, images[index])}))
        return dict(model=dict(self.identity), predictions=predictions, inference_seconds=time.monotonic() - started,
            action_units=list(self.identity["action_units"]), action_horizon=7, smoothing="none", native_action_dim=6,
            fixed_gripper_full_width_m=self.identity["fixed_gripper_full_width_m"])
