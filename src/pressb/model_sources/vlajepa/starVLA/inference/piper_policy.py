"""Strict checkpoint inference for PiPER's absolute base_link TCP pose policy.

The network predicts xyz (metres) + the first two ROWS of R, with no scaling.
The optional 8D output reconstructs unit wxyz and appends an explicit fixed
controller gripper opening; that eighth value is never a learned prediction.
This module does not run IK or execute robot commands.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image


CAMERA_KEYS = ("observation.images.global", "observation.images.wrist")
CONTROLLER_DEFAULT_GRIPPER_WIDTH_M = 0.008
_SCHEMA = dict(state_dim=9, action_dim=9, action_horizon=7, video_horizon=8,
               fps=30, pose_frame="base_link", pose_link="gripper_tcp",
               xyz_units="metres", normalization="none",
               rotation="R[:2,:].reshape(6); first two rows",
               gripper="excluded; fixed by controller",
               episode_end="first sampled target light on",
               terminal_action="repeat previous planned target")


def _pose_helpers():
    # Lazy import keeps schema/checksum inspection independent of model imports.
    from starVLA.dataloader import piper_lerobot
    return piper_lerobot


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_action_schema(schema):
    """Reject conventions that would reinterpret a compatible-looking tensor."""
    for key, expected in _SCHEMA.items():
        if schema.get(key) != expected or type(schema.get(key)) is not type(expected):
            raise ValueError(f"Unsupported action schema {key}: expected {expected!r}, got {schema.get(key)!r}")
    for key in ("dataset_sha256", "split_sha256", "training_fingerprint"):
        if not isinstance(schema.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", schema[key]):
            raise ValueError(f"Missing/invalid action schema provenance: {key}")
    return dict(schema)


def validate_checkpoint(checkpoint):
    """Verify complete checkpoint weights and binding to run metadata.

    best/last symlinks are resolved once, so a later training save cannot switch
    the checkpoint between validation and loading. Optimizer state is unused.
    """
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    if checkpoint.name == "model.pt" and checkpoint.is_file():
        checkpoint = checkpoint.parent
    if not checkpoint.is_dir() or checkpoint.parent.name != "checkpoints":
        raise ValueError("Expected run/checkpoints/{best,last,step_*} or its model.pt")
    run = checkpoint.parent.parent
    manifest = _read_json(checkpoint / "manifest.json")
    if manifest.get("complete") is not True or type(manifest.get("step")) is not int or manifest["step"] < 1:
        raise ValueError("Checkpoint must be complete and contain at least one training step")
    model_sha = sha256(checkpoint / "model.pt")
    if model_sha != manifest.get("model_sha256"):
        raise ValueError("Checkpoint model.pt SHA256 mismatch")
    schema = validate_action_schema(_read_json(run / "action_schema.json"))
    for key in ("dataset_sha256", "split_sha256", "training_fingerprint"):
        if manifest.get(key) != schema[key]:
            raise ValueError(f"Checkpoint/action schema {key} mismatch")
    split = _read_json(run / "split.json")
    if canonical_sha256(split) != schema["split_sha256"]:
        raise ValueError("Split canonical SHA256 mismatch")
    if split.get("source_export_manifest_sha256") != schema["dataset_sha256"]:
        raise ValueError("Split dataset identity mismatch")
    if split.get("camera_keys") != list(CAMERA_KEYS):
        raise ValueError("Camera order must be global, wrist")
    contract = _read_json(run / "training_contract.json")
    if (contract.get("fingerprint") != schema["training_fingerprint"]
            or canonical_sha256(contract.get("contract")) != schema["training_fingerprint"]):
        raise ValueError("Training contract fingerprint mismatch")
    if not (run / "processor").is_dir() or not (run / "config.yaml").is_file():
        raise ValueError("Run must include its saved processor/ and config.yaml")
    return dict(run=run, checkpoint=checkpoint, manifest=manifest, schema=schema,
                split=split, contract=contract["contract"],
                provenance=dict(checkpoint=str(checkpoint), model_sha256=model_sha,
                                manifest_sha256=sha256(checkpoint / "manifest.json"),
                                config_sha256=sha256(run / "config.yaml"),
                                action_schema_sha256=sha256(run / "action_schema.json"),
                                dataset_sha256=schema["dataset_sha256"],
                                split_sha256=schema["split_sha256"],
                                training_fingerprint=schema["training_fingerprint"],
                                processor_files_sha256={str(p.relative_to(run / "processor")): sha256(p)
                                    for p in sorted((run / "processor").rglob("*")) if p.is_file()}))


def validate_config_contract(config, expected):
    """Match the trainer's saved contract before permitting device/path changes."""
    framework = copy.deepcopy(config["framework"])
    for key in ("device_map", "base_vlm"):
        framework["qwenvl"].pop(key, None)
    framework["vj2_model"].pop("base_encoder", None)
    actual = dict(seed=config["seed"], framework=framework,
                  data={key: config["datasets"]["vla_data"][key] for key in expected["data"]},
                  trainer={key: config["trainer"][key] for key in expected["trainer"]})
    if actual != expected:
        raise ValueError("config.yaml differs from the saved training contract")
    if (config["framework"]["qwenvl"].get("init_from_config") is not True
            or config["framework"]["vj2_model"].get("init_from_config") is not True):
        raise ValueError("PiPER inference requires init_from_config=true for both backbones")
    if not isinstance(config["datasets"]["vla_data"]["resolution_size"], int) or config["datasets"]["vla_data"]["resolution_size"] < 1:
        raise ValueError("resolution_size must be a positive integer")


def pose9_to_pose8(pose, *, controller_gripper_width_m=CONTROLLER_DEFAULT_GRIPPER_WIDTH_M):
    """Project predicted rot6 to SO(3), preserving xyz; append fixed opening.

    Zero/parallel rotation rows raise ValueError. No position clipping, unit
    conversion, frame conversion, normalization or IK is applied.
    """
    values = np.asarray(pose)
    if values.shape[-1:] != (9,) or not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("Expected finite floating (...,9) xyz + first-two-rows rotation")
    width = float(controller_gripper_width_m)
    if not np.isfinite(width) or width < 0:
        raise ValueError("Controller gripper width must be finite and nonnegative, in metres")
    result = np.empty(values.shape[:-1] + (8,), dtype=values.dtype)
    result[..., :3] = values[..., :3]
    result[..., 3:7] = _pose_helpers().rotation_6d_to_quaternion_wxyz(values[..., 3:])
    result[..., 7] = width
    return result


def prepare_state(state):
    """Accept one measured raw8 or pose9 state; return float32 [1,1,9]."""
    value = np.asarray(state, dtype=np.float32)
    if value.shape not in ((8,), (9,)) or not np.isfinite(value).all():
        raise ValueError("state must be one finite 8D or 9D vector, without batch dimensions")
    helpers = _pose_helpers()
    if value.shape == (8,):
        value = helpers.pose8_to_pose9(value)
    else:
        rotation = helpers.rotation_6d_to_matrix(value[3:])
        if not np.allclose(rotation[:2].reshape(6), value[3:], atol=1e-5, rtol=0):
            raise ValueError("Measured state9 must contain the first two orthonormal rows of R")
    return value.reshape(1, 1, 9).copy()


def load_trained_state(model, model_path):
    """Plain full state_dict only: no pretrained allowlist or partial loading."""
    import torch
    state = torch.load(model_path, map_location="cpu", weights_only=True, mmap=True)
    if (not isinstance(state, dict) or not state
            or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items())
            or any(not any(k.startswith(prefix) for k in state) for prefix in
                   ("qwen_vl_interface.", "vj_encoder.", "vj_predictor.", "action_model."))):
        raise ValueError("Expected a plain complete four-component trained model state_dict")
    # strict=True checks names/shapes but would silently overwrite aliases of
    # a tied parameter in sequence. Their saved values must agree beforehand.
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    for names in aliases.values():
        if len(names) > 1 and (any(name not in state for name in names)
                              or any(state[names[0]].dtype != state[name].dtype
                                     or not torch.equal(state[names[0]], state[name]) for name in names[1:])):
            raise ValueError(f"Trained checkpoint contains inconsistent tied parameter aliases: {names}")
    # The constructor creates Qwen in BF16, whereas fine-tuning stores its
    # trainable parameters in FP32. A plain strict copy otherwise rounds those
    # trained tensors to BF16. Preserve source dtypes and existing tied objects.
    for name, tensor in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False)):
        if name in state and tensor.dtype != state[name].dtype:
            tensor.data = tensor.data.to(dtype=state[name].dtype)
    model.load_state_dict(state, strict=True)
    return len(state)


def install_saved_processor(model, processor):
    """Prevent silently changing token IDs when moving a run to another host."""
    constructed = model.qwen_vl_interface.processor.tokenizer
    saved = processor.tokenizer
    if constructed.get_vocab() != saved.get_vocab():
        raise ValueError("Saved processor tokenizer vocabulary differs from constructed architecture")
    for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
        if getattr(constructed, name, None) != getattr(saved, name, None):
            raise ValueError(f"Saved processor tokenizer {name} differs")
    saved.padding_side = "left"
    model.qwen_vl_interface.processor = processor


def disable_unused_qwen_cache(model):
    """Match training validation: full image/text inputs need no decoder KV cache."""
    config = model.qwen_vl_interface.model.config
    config.use_cache = False
    if hasattr(config, "text_config"):
        config.text_config.use_cache = False


class PiperPolicy:
    """One local CUDA model using its verified trained checkpoint and processor."""

    def __init__(self, checkpoint, *, device="cuda:0",
                 controller_gripper_width_m=CONTROLLER_DEFAULT_GRIPPER_WIDTH_M,
                 base_vlm=None, base_encoder=None):
        self._weights_loaded = False
        checked = validate_checkpoint(checkpoint)
        from omegaconf import OmegaConf
        import torch
        from transformers import AutoProcessor
        cfg = OmegaConf.load(checked["run"] / "config.yaml")
        validate_config_contract(OmegaConf.to_container(cfg, resolve=True), checked["contract"])
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("PiperPolicy model execution requires an available local CUDA device")
        torch.cuda.set_device(self.device)
        cfg.framework.qwenvl.device_map = str(self.device)
        for section, key, override in ((cfg.framework.qwenvl, "base_vlm", base_vlm),
                                       (cfg.framework.vj2_model, "base_encoder", base_encoder)):
            path = Path(override if override is not None else section[key]).expanduser().resolve()
            if not (path / "config.json").is_file():
                raise ValueError(f"Missing local backbone configuration: {path}; provide its path override")
            section[key] = str(path)
            section.init_from_config = True
        self.controller_gripper_width_m = float(controller_gripper_width_m)
        if not np.isfinite(self.controller_gripper_width_m) or self.controller_gripper_width_m < 0:
            raise ValueError("Controller gripper width must be finite and nonnegative")
        self.resolution = int(cfg.datasets.vla_data.resolution_size)
        self.config, self.schema = cfg, checked["schema"]
        self.run, self.split = checked["run"], checked["split"]
        self.provenance = checked["provenance"]
        from starVLA.model.framework.VLA_JEPA import VLA_JEPA
        model = VLA_JEPA(config=cfg)
        processor = AutoProcessor.from_pretrained(self.run / "processor", local_files_only=True)
        install_saved_processor(model, processor)
        self.provenance["loaded_state_tensors"] = load_trained_state(model, checked["checkpoint"] / "model.pt")
        disable_unused_qwen_cache(model)
        model.to(self.device).requires_grad_(False).eval()
        self.model = model
        self._weights_loaded = True

    def predict(self, *, global_image, wrist_image, state, task, seed=None):
        """Return raw_pose9 [7,9] and pose8 [7,8], both absolute base_link TCP."""
        if not self._weights_loaded:
            raise RuntimeError("Prediction forbidden before strict trained-weight loading")
        if not isinstance(task, str) or not re.fullmatch(r"Press (?:2[4-9]|3[0-5]) floor\.", task):
            raise ValueError("Expected an exact trained task: 'Press xx floor.', for floors 24..35")
        images = []
        for image in (global_image, wrist_image):
            if not isinstance(image, Image.Image):
                raise TypeError("global_image and wrist_image must be PIL images")
            images.append(image.convert("RGB").resize((self.resolution, self.resolution), Image.Resampling.BILINEAR))
        measured = prepare_state(state)
        import torch
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        rng_scope = torch.random.fork_rng(devices=devices) if seed is not None else nullcontext()
        self.model.eval()
        with rng_scope, torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            if seed is not None:
                torch.manual_seed(int(seed))
            output = self.model.predict_action(batch_images=[images], instructions=[task], state=measured)
        # Historical upstream key; this PiPER run uses normalization='none'.
        raw = np.asarray(output["normalized_actions"], dtype=np.float32)
        if raw.shape != (1, 7, 9) or not np.isfinite(raw).all():
            raise ValueError("Model must return finite [1,7,9] raw poses")
        raw = raw[0].copy()
        pose8 = pose9_to_pose8(raw, controller_gripper_width_m=self.controller_gripper_width_m)
        return dict(raw_pose9=raw, pose8=pose8,
                    controller_gripper_width_m=self.controller_gripper_width_m,
                    gripper_is_learned=False, pose_frame="base_link", pose_link="gripper_tcp", xyz_units="metres")


def _comparison(predicted, target, action_is_pad):
    valid = ~np.asarray(action_is_pad, dtype=bool)
    p = _pose_helpers()
    relative = p.rotation_6d_to_matrix(predicted[:, 3:]) @ np.swapaxes(p.rotation_6d_to_matrix(target[:, 3:]), -1, -2)
    angle = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1., 1.))
    return dict(valid_target_steps=int(valid.sum()),
                xyz_mae_m=float(np.abs(predicted[valid, :3] - target[valid, :3]).mean()),
                rotation_geodesic_mean_deg=float(np.rad2deg(angle[valid]).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-episode", type=int, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-vlm", type=Path)
    parser.add_argument("--base-encoder", type=Path)
    parser.add_argument("--controller-gripper-width-m", type=float, default=CONTROLLER_DEFAULT_GRIPPER_WIDTH_M)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite inference report: {args.output}")
    policy = PiperPolicy(args.checkpoint, device=args.device, base_vlm=args.base_vlm, base_encoder=args.base_encoder,
                         controller_gripper_width_m=args.controller_gripper_width_m)
    split = next((name for name in ("train", "val") if args.source_episode in policy.split[f"{name}_source_episode_ids"]), None)
    if split is None:
        raise ValueError("Source episode is absent from this trained run's split")
    dataset = _pose_helpers().PiperLeRobotDataset(args.dataset_root, split=split, split_manifest=policy.split,
                resolution=policy.resolution, seed=int(policy.config.seed))
    try:
        if dataset.export_manifest_sha256 != policy.schema["dataset_sha256"]:
            raise ValueError("Dataset differs from the trained run")
        sample = dataset.get_frame(args.source_episode, args.frame)
        prediction = policy.predict(global_image=sample["image"][0], wrist_image=sample["image"][1],
                                    state=sample["state"][0], task=sample["lang"], seed=args.seed)
        report = dict(kind="piper_trained_policy_offline_prediction", schema_version=1,
                      provenance=policy.provenance, policy_code_sha256=sha256(__file__), seed=args.seed,
                      source_episode_id=args.source_episode, frame_index=args.frame, split=split,
                      task=sample["lang"], camera_keys=list(CAMERA_KEYS), state9=sample["state"][0].tolist(),
                      target_pose9=sample["action"].tolist(), action_is_pad=sample["action_is_pad"].tolist(),
                      target_semantics="stored action[i:i+7]; terminal holds last executed planned target; out-of-range rows clamp and are padded",
                      metrics=_comparison(prediction["raw_pose9"], sample["action"], sample["action_is_pad"]),
                      **{key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in prediction.items()})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(dict(output=str(args.output.resolve()), metrics=report["metrics"])))
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
