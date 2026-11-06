"""Strict joint7 policies with explicit per-sample noise and batched inference."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

COLORS = ("red", "yellow", "blue")
SCALE = np.array([180.] * 6 + [8.], dtype=np.float32)
CAMERAS = ("up", "wrist")
SCHEMA = "pressb.table_buttons.joint7_policy.v1"


class InvalidPolicyAction(RuntimeError):
    """A model failure to be counted as an episode outcome, not a transport retry."""


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_state(state):
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("state must be finite joint1..6 degrees and full gripper width centimetres")
    return state / SCALE


def prepare_images(images, resolution=224):
    if len(images) != 2:
        raise ValueError("Expected up and wrist cameras in that order")
    result = []
    for value in images:
        if not isinstance(value, Image.Image) or value.mode != "RGB" or value.size != (640, 480):
            raise ValueError("Input cameras must be RGB 640 x 480")
        result.append(value.resize((resolution, resolution), Image.Resampling.BILINEAR))
    return result


def load_metadata(run, model_id):
    from omegaconf import OmegaConf
    run = Path(run).resolve(strict=True)
    archive = read_json(run / "archive.json")
    schema, normalization = read_json(run / "action_schema.json"), read_json(run / "normalization.json")
    config = OmegaConf.load(run / "config.yaml")
    best_path = (run / "checkpoints/best/model.pt").resolve(strict=True)
    manifest = read_json(best_path.parent / "manifest.json")
    if archive["group"] != model_id or config.run_id != model_id:
        raise ValueError("Requested model identity differs from run metadata")
    if not Path(archive["model_path"]).samefile(best_path):
        raise ValueError("Best alias differs from archive model")
    if best_path.stat().st_size != archive["model_bytes"] or archive["model_bytes"] != manifest["model_bytes"]:
        raise ValueError("Checkpoint size differs from archive")
    if manifest.get("complete") is not True or manifest["step"] != archive["best_step"]:
        raise ValueError("Best checkpoint is incomplete or has wrong step")
    for key in ("state_dim", "action_dim", "action_horizon"):
        if schema[key] != 7:
            raise ValueError(f"Expected joint7 schema: {key}")
    if schema["cameras"] != list(CAMERAS) or schema["fps"] != 30:
        raise ValueError("Camera/timing schema mismatch")
    if (normalization["scale"] != SCALE.tolist() or normalization["offset"] != [0.] * 7
            or normalization["clip"] is not False or normalization["gripper_representation"] != "full_width"):
        raise ValueError("Joint7 fixed normalization mismatch")
    if schema["normalization"] != normalization:
        raise ValueError("Schema and normalization disagree")
    for key in ("dataset_sha256", "split_sha256", "training_fingerprint"):
        if schema[key] != manifest[key] or archive[key] != manifest[key]:
            raise ValueError(f"Checkpoint provenance mismatch: {key}")
    spacing, task = model_id.split("_", 1)
    if spacing not in ("near", "medium", "far") or task not in (*COLORS, "multitask"):
        raise ValueError("Invalid table-buttons model ID")
    identity = dict(model_id=model_id, spacing=spacing, task_type="multitask" if task == "multitask" else "single",
                    allowed_instructions=[f"Press the {color} button." for color in (COLORS if task == "multitask" else (task,))],
                    model_sha256=archive["model_sha256"], model_bytes=archive["model_bytes"], best_step=archive["best_step"],
                    checkpoint_path=str(best_path), run=str(run), camera_order=list(CAMERAS),
                    image_input_size=[640, 480], image_resize="PIL RGB BILINEAR square; identical to training adapter",
                    image_model_size=[int(config.datasets.vla_data.resolution_size)] * 2,
                    state_units=["degree"] * 6 + ["full_width_cm"], action_units=["degree"] * 6 + ["full_width_cm"],
                    normalization_scale=SCALE.tolist(), action_horizon=7, action_dim=7, fps=30,
                    smoothing="none", action_clipping="none", recurrent_state="none",
                    num_inference_timesteps=int(config.framework.action_model.num_inference_timesteps))
    return config, identity


def seeded_noise(head, batch_size, seeds, device):
    """Each row matches a fresh torch.manual_seed(seed) single-sample draw."""
    import torch
    if len(seeds) != batch_size or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
        raise ValueError("One integer noise seed in [0, 2**63) is required for each sample")
    rows = [torch.randn((1, head.action_horizon, head.action_dim), dtype=head.dtype, device=device,
                       generator=torch.Generator(device=device).manual_seed(seed)) for seed in seeds]
    return torch.cat(rows, dim=0)


def flow_with_noise(head, embeddings, state, noise):
    """Upstream FlowmatchingActionHead Euler solver with explicit input noise.

    No class is patched. This adapter changes only noise construction; every
    encoding, timestep and update matches the existing inference head.
    """
    import torch
    embeddings = embeddings.to(dtype=head.dtype)
    batch = embeddings.shape[0]
    actions = noise
    steps = head.num_inference_timesteps
    if steps is None or steps < 1:
        raise ValueError("Invalid inference step count")
    state_features = head._encode_state(state, batch)
    for step in range(steps):
        times = torch.full((batch,), int(step / float(steps) * head.num_timestep_buckets), device=embeddings.device)
        action_features = head.action_encoder(actions, times)
        if head.config.add_pos_embed:
            position_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=embeddings.device)
            action_features = action_features + head.position_embedding(position_ids).unsqueeze(0)
        future = head.future_tokens.weight.unsqueeze(0).expand(batch, -1, -1)
        tokens = torch.cat((state_features, future, action_features), dim=1) if state_features is not None else torch.cat((future, action_features), dim=1)
        output = head.model(hidden_states=tokens, encoder_hidden_states=embeddings, timestep=times)
        velocity = head.action_decoder(output)[:, -head.action_horizon:]
        actions = actions + (1.0 / steps) * velocity
    return actions


class TableButtonsPolicy:
    def __init__(self, run, model_id, device="cuda:0"):
        import torch
        from transformers import AutoProcessor
        from starVLA.model.framework.VLA_JEPA import VLA_JEPA
        from starVLA.training import train_piper as shared
        from starVLA.inference.piper_policy import install_saved_processor
        started = time.monotonic()
        self.config, self.identity = load_metadata(run, model_id)
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.config.framework.qwenvl.device_map = "cpu"
        if not self.config.framework.qwenvl.init_from_config or not self.config.framework.vj2_model.init_from_config:
            raise ValueError("Expected architecture-only construction before strict load")
        if sha256(self.identity["checkpoint_path"]) != self.identity["model_sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        shared.seed_all(42)
        self.model = VLA_JEPA(config=self.config)
        install_saved_processor(self.model, AutoProcessor.from_pretrained(Path(run) / "processor", local_files_only=True))
        shared.configure_parameters(self.model, checkpointing=False)
        weights = torch.load(self.identity["checkpoint_path"], map_location="cpu", weights_only=True, mmap=True)
        shared.validate_tied_source(self.model, weights)
        if set(weights) != set(self.model.state_dict()):
            raise ValueError("Model and checkpoint keys differ")
        for name, tensor in list(self.model.named_parameters(remove_duplicate=False)) + list(self.model.named_buffers(remove_duplicate=False)):
            if name in weights and tensor.dtype != weights[name].dtype:
                tensor.data = tensor.data.to(weights[name].dtype)
        self.model.load_state_dict(weights, strict=True)
        if any(value.dtype != weights[name].dtype for name, value in self.model.state_dict().items()):
            raise ValueError("Strict loading changed saved tensor dtype")
        self.identity.update(strict_loaded=True, model_digest_verified=True, loaded_keys=len(weights),
                             saved_dtype_counts=dict(Counter(str(value.dtype) for value in weights.values())))
        del weights
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(self.device)
        self.config.framework.qwenvl.device_map = str(self.device)
        self.resolution = int(self.config.datasets.vla_data.resolution_size)
        self.identity.update(device=str(self.device), loaded_seconds=time.monotonic() - started)

    def prepare(self, samples):
        images, states, instructions, seeds = [], [], [], []
        for sample in samples:
            instruction = sample["instruction"]
            if instruction not in self.identity["allowed_instructions"]:
                raise ValueError(f"Instruction {instruction!r} is not allowed for {self.identity['model_id']}")
            images.append(prepare_images(sample["images"], self.resolution))
            states.append(normalize_state(sample["state"])[None])
            instructions.append(instruction)
            seeds.append(sample["noise_seed"])
        return images, np.stack(states), instructions, seeds

    def predict_prepared(self, images, states, instructions, seeds):
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
            normalized = actions.float().cpu().numpy()
            if normalized.shape != (len(images), 7, 7) or not np.isfinite(normalized).all():
                raise InvalidPolicyAction("Nonfinite or incorrectly shaped generated actions")
            return normalized, [hashlib.sha256(row.float().cpu().numpy().tobytes()).hexdigest() for row in noise]

    def predict(self, samples):
        import torch
        started = time.monotonic()
        images, states, instructions, seeds = self.prepare(samples)
        normalized, noise_hashes = self.predict_prepared(images, states, instructions, seeds)
        physical = normalized * SCALE
        torch.cuda.synchronize(self.device)
        predictions = []
        for index, sample in enumerate(samples):
            predictions.append(dict(episode_id=sample["episode_id"], request_id=sample["request_id"], noise_seed=seeds[index],
                actions=physical[index].tolist(), normalized_actions=normalized[index].tolist(),
                normalized_state=states[index, 0].tolist(), initial_noise_sha256=noise_hashes[index],
                preprocessed_image_sha256={camera: hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
                                          for camera, image in zip(CAMERAS, images[index])}))
        return dict(model=dict(self.identity), predictions=predictions, inference_seconds=time.monotonic() - started,
                    action_units=list(self.identity["action_units"]), action_horizon=7, smoothing="none")
