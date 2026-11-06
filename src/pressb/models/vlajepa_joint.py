"""Frozen joint6/joint7 VLA-JEPA inference with SI runtime boundaries.

The checkpoint retains its degree/full-width-centimetre normalization. The
runtime receives radians/metres, and joint6 retains a six-dimensional action
and noise head. The task controller owns its declared fixed gripper command.
No HTTP-only policy is presented as an encoder or an external-noise decoder.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
from PIL import Image

from pressb.model_sources import activate_source, builtin_source_root, source_tree_sha256
from pressb.runtime import contract as runtime_contract, image_transport
from pressb.runtime.contract import decode_image, finite_array, fingerprint, validate_observation
from pressb.runtime.image_transport import ValidatedObservation


NAMES = [f"joint{i}" for i in range(1, 7)] + ["gripper_width"]
UNITS = ["rad"] * 6 + ["m"]
MODEL_SCALE = np.asarray([180.] * 6 + [8.], dtype=np.float32)
SI_TO_MODEL = np.asarray([180. / np.pi] * 6 + [100.], dtype=np.float32)
MODEL_TO_SI = np.asarray([np.pi / 180.] * 6 + [.01], dtype=np.float32)
FIXED_GRIPPER_WIDTH_M = float(np.float32(.02)) * .01


def _read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_joint_schema(schema, normalization):
    """Fail before model loading on ambiguous dimensions, units or scaling."""
    dimension = schema.get("action_dim")
    if type(dimension) is not int or dimension not in (6, 7):
        raise ValueError("Joint VLA-JEPA requires a native joint6 or joint7 action head")
    expected = dict(schema=f"vlajepa.table_buttons.joint{dimension}.v1", state_dim=7,
                    action_horizon=7, video_horizon=8, fps=30, cameras=["up", "wrist"],
                    dataset_units=["deg"] * 6 + ["cm"])
    for key, value in expected.items():
        if schema.get(key) != value or type(schema.get(key)) is not type(value):
            raise ValueError(f"Unsupported joint checkpoint contract: {key}")
    if schema.get("normalization") != normalization:
        raise ValueError("Action schema and normalization.json differ")
    if dimension == 6:
        if normalization.get("kind") != "separate_joint7_state_joint6_action" or normalization.get("version") != 1:
            raise ValueError("joint6 requires separate joint7-state/joint6-action normalization")
        state, action = normalization.get("state", {}), normalization.get("action", {})
        width = normalization.get("fixed_gripper_full_width_m")
        if (type(width) not in (int, float) or width != FIXED_GRIPPER_WIDTH_M
                or schema.get("fixed_gripper_full_width_m") != width):
            raise ValueError("joint6 fixed gripper differs from the source full-width centimetre contract")
    else:
        state = action = normalization
    for spec, size in ((state, 7), (action, dimension)):
        expected = dict(kind=f"fixed_joint{size}_scale", version=1,
                        feature_names=[f"joint_{i}.pos" for i in range(1, 7)] + (["gripper.pos"] if size == 7 else []),
                        input_units=["degree"] * 6 + (["cm"] if size == 7 else []),
                        scale=MODEL_SCALE[:size].tolist(), offset=[0.] * size, clip=False)
        for key, value in expected.items():
            if spec.get(key) != value or type(spec.get(key)) is not type(value):
                raise ValueError(f"Invalid joint{size} normalization: {key}")
        if size == 7 and spec.get("gripper_representation") != "full_width":
            raise ValueError("Gripper normalization must represent full width")
    return dimension


def read_checkpoint_metadata(run, checkpoint=None, *, model_id=None):
    """Validate saved contracts without reading weights or starting CUDA.

    Full model checksumming and strict tensor loading are mandatory in the
    factory. This inspection function deliberately does neither operation.
    """
    import yaml

    run = Path(run).expanduser().resolve(strict=True)
    checkpoint = Path(checkpoint).expanduser() if checkpoint is not None else run / "checkpoints/best"
    checkpoint = checkpoint.resolve(strict=True)
    if checkpoint.is_file() and checkpoint.name == "model.pt":
        checkpoint = checkpoint.parent
    model_path = checkpoint / "model.pt"
    schema = _read_json(run / "action_schema.json")
    normalization = _read_json(run / "normalization.json")
    dimension = validate_joint_schema(schema, normalization)
    manifest = _read_json(checkpoint / "manifest.json")
    config = yaml.safe_load((run / "config.yaml").read_text())
    if not isinstance(config, dict) or not isinstance(config.get("run_id"), str):
        raise ValueError("Checkpoint config must identify its run")
    if model_id is not None and model_id != config["run_id"]:
        raise ValueError("model_id differs from the saved run_id")
    if (manifest.get("complete") is not True or manifest.get("schema") != schema["schema"]
            or type(manifest.get("step")) is not int or manifest["step"] < 1
            or model_path.stat().st_size != manifest.get("model_bytes")):
        raise ValueError("Checkpoint is incomplete or its schema/size differs")
    for key in ("dataset_sha256", "split_sha256", "training_fingerprint"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(schema.get(key, ""))) or manifest.get(key) != schema[key]:
            raise ValueError(f"Checkpoint provenance mismatch: {key}")
    split = _read_json(run / "split.json")
    if (fingerprint(split) != schema["split_sha256"]
            or split.get("source_export_manifest_sha256") != schema["dataset_sha256"]
            or split.get("camera_keys") != ["observation.images.up", "observation.images.wrist"]):
        raise ValueError("Split digest, dataset identity or camera order differs")
    training = _read_json(run / "training_contract.json")
    expected = training.get("contract", {})
    if training.get("fingerprint") != schema["training_fingerprint"] or fingerprint(expected) != schema["training_fingerprint"]:
        raise ValueError("Training contract fingerprint differs")
    framework = deepcopy(config["framework"])
    framework["qwenvl"].pop("device_map", None)
    framework["qwenvl"].pop("base_vlm", None)
    framework["vj2_model"].pop("base_encoder", None)
    actual = dict(seed=config["seed"], framework=framework,
                  data={key: config["datasets"]["vla_data"][key] for key in expected["data"]},
                  trainer={key: config["trainer"][key] for key in expected["trainer"]})
    if any(actual[key] != expected[key] for key in actual) or expected.get("normalization") != normalization:
        raise ValueError("Saved config differs from its training contract")
    for key, value in dict(state_dim=7, action_dim=dimension, action_horizon=7).items():
        if config["framework"]["action_model"].get(key) != value:
            raise ValueError(f"Joint model architecture differs: {key}")
    if any(config["framework"][name].get("init_from_config") is not True for name in ("qwenvl", "vj2_model")):
        raise ValueError("Joint inference requires architecture-only construction before strict loading")
    resolution = config["datasets"]["vla_data"]["resolution_size"]
    if type(resolution) is not int or resolution < 1 or not (run / "processor").is_dir():
        raise ValueError("Checkpoint requires a saved processor and positive resolution")
    colors = schema.get("target_colors", expected.get("target_colors"))
    if colors is None:
        group = schema.get("dataset_group", config["run_id"])
        colors = [group.split("_")[1]] if "_" in group else []
    if not isinstance(colors, list) or not colors or any(color not in ("red", "yellow", "blue") for color in colors):
        raise ValueError("Checkpoint must declare its trained target colors")
    # Published hashes can live in the checkpoint or in portable archive
    # receipts. Archive host paths are not authoritative after relocation.
    digests = [manifest["model_sha256"]] if manifest.get("model_sha256") else []
    archive_path = run / "archive.json"
    if archive_path.is_file():
        archive = _read_json(archive_path)
        if archive.get("best_step") == manifest["step"]:
            if archive.get("model_bytes") != model_path.stat().st_size:
                raise ValueError("Archive size differs from checkpoint")
            digests.append(archive.get("model_sha256"))
    durable_path = run / "durable_archive.json"
    if durable_path.is_file():
        receipt = _read_json(durable_path)
        for alias, digest in receipt.get("checkpoint_model_sha256", {}).items():
            candidate = run / "checkpoints" / alias / "model.pt"
            if candidate.is_file() and candidate.samefile(model_path):
                digests.append(digest)
    if any(not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in digests) or len(set(digests)) > 1:
        raise ValueError("Published checkpoint digests are malformed or disagree")
    return dict(run=run, checkpoint=checkpoint, model_path=model_path, config=config, schema=schema,
                normalization=normalization, manifest=manifest, action_dim=dimension,
                expected_sha256=digests[0] if digests else None,
                allowed_instructions=[f"Press the {color} button." for color in colors])


@dataclass(frozen=True)
class JointContext:
    tokens: object
    state: object
    baseline_noise: object


class JointVLAJEPABackend:
    """Split encoder/flow decoder over a strictly loaded frozen native model."""

    def __init__(self, policy, *, instruction_pattern=None):
        self.policy = policy
        self.model = policy.model
        self.dimension = validate_joint_schema(policy.schema, policy.normalization)
        head = self.model.action_model
        if (head.action_dim != self.dimension or head.action_horizon != 7 or head.config.state_dim != 7):
            raise ValueError("Loaded model does not match native action/state dimensions")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("Joint VLA-JEPA must be frozen before adapting it")
        self.horizon = head.action_horizon
        self.feature_dim = int(self.model.qwen_vl_interface.model.config.hidden_size)
        self.token_count = int(self.model.config.framework.vj2_model.num_embodied_action_tokens_per_instruction)
        if self.feature_dim < 1 or self.token_count < 1:
            raise ValueError("Invalid embodied-token feature contract")
        if instruction_pattern is not None and (not isinstance(instruction_pattern, str) or not instruction_pattern):
            raise ValueError("instruction_pattern must be a nonempty expression or null")
        self.instruction_pattern = None if instruction_pattern is None else re.compile(instruction_pattern)
        self._metadata = dict(schema_version=2, node_type="inference", model_id=policy.identity["model_id"],
            capabilities=["encode", "decode", "infer", "external_initial_noise", "features", "context_cache"],
            observation=dict(state=dict(names=NAMES.copy(), units=UNITS.copy()), cameras=["up", "wrist"]),
            action=dict(representation="joint_position", frame="robot_base", names=NAMES[:self.dimension],
                        units=UNITS[:self.dimension], horizon=self.horizon, control_hz=30),
            noise=dict(shape=[self.horizon, self.dimension], semantics="flow_initial_state",
                       baseline="per_observation_seeded_gaussian"),
            feature_dim=self.feature_dim, feature_kind="mean_embodied_action_tokens",
            frozen=True, native_action_dim=self.dimension, state_dim=7,
            gripper_learned=self.dimension == 7, allowed_instructions=list(policy.allowed_instructions),
            instruction_pattern=instruction_pattern, normalization=deepcopy(policy.normalization),
            checkpoint_sha256=policy.identity["model_sha256"], checkpoint_step=policy.identity["checkpoint_step"],
            source_revision=policy.identity["source_revision"],
            model_config_sha256=fingerprint(dict(config=policy.identity["config_sha256"],
                schema=policy.schema, instruction_pattern=instruction_pattern)),
            normalization_sha256=fingerprint(policy.normalization), seed_scope="per_observation",
            context_cache_kind="embodied_action_tokens_and_normalized_joint_state",
            batching=dict(native_model_batching=True), image_encodings=["png_base64", "png_bytes"], image_mode="RGB")
        if self.dimension == 6:
            self._metadata["fixed_gripper_width_m"] = policy.normalization["fixed_gripper_full_width_m"]

    def describe(self):
        return deepcopy(self._metadata)

    def _prepare(self, observation):
        validate_observation(observation, self._metadata["observation"])
        instruction = observation["task"]
        if instruction not in self.policy.allowed_instructions:
            raise ValueError("Instruction is outside the checkpoint's trained task contract")
        if self.instruction_pattern is not None and not self.instruction_pattern.fullmatch(instruction):
            raise ValueError("Instruction does not match instruction_pattern")
        if observation.get("model_inputs"):
            raise ValueError("Joint VLA-JEPA does not accept extra model_inputs")
        state = finite_array(observation["state"], (7,), name="state") * SI_TO_MODEL / MODEL_SCALE
        images = []
        for camera in ("up", "wrist"):
            image = (observation.decoded_images[camera] if isinstance(observation, ValidatedObservation)
                     else decode_image(observation["images"][camera], require_rgb=True))
            if image.mode != "RGB" or image.size != (640, 480):
                raise ValueError("Joint checkpoint cameras must be RGB 640 x 480")
            images.append(image.resize((self.policy.resolution, self.policy.resolution), Image.Resampling.BILINEAR))
        return images, state[None], instruction

    def validate_observations(self, observations):
        for observation in observations:
            self._prepare(observation)

    def encode_batch(self, observations, seeds):
        import torch

        if len(observations) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
            raise ValueError("Expected one integer seed in [0, 2**63) per observation")
        if not observations:
            return []
        prepared = [self._prepare(observation) for observation in observations]
        model = self.model
        model.eval()
        with torch.inference_mode():
            inputs = model.qwen_vl_interface.build_qwenvl_inputs(
                images=[row[0] for row in prepared], instructions=[row[2] for row in prepared],
                prompt_replace_dict={"{actions}": model.replace_prompt, "{e_actions}": model.embodied_replace_prompt})
            ids = inputs["input_ids"]
            mask = ids == model.embodied_action_token_id
            if ids.shape[0] != len(prepared) or not torch.all(mask.sum(dim=1) == self.token_count):
                raise ValueError("Processor changed the trained embodied-action token count")
            device_type = self.policy.device.type
            with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
                output = model.qwen_vl_interface(**inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
                tokens = output.hidden_states[-1][mask].reshape(len(prepared), self.token_count, self.feature_dim)
            tokens = tokens.to(dtype=model.action_model.dtype)
            states = torch.as_tensor(np.stack([row[1] for row in prepared]), device=tokens.device, dtype=model.action_model.dtype)
            if not torch.isfinite(tokens).all() or not torch.isfinite(states).all() or states.shape != (len(prepared), 1, 7):
                raise ValueError("Invalid frozen joint encoder context")
            features = tokens.float().mean(dim=1).cpu().numpy()
            result = []
            for index, seed in enumerate(seeds):
                baseline = torch.randn((1, self.horizon, self.dimension), device=tokens.device, dtype=tokens.dtype,
                    generator=torch.Generator(device=tokens.device).manual_seed(seed))
                context = JointContext(tokens[index:index + 1].clone(), states[index:index + 1].clone(), baseline)
                feature = finite_array(features[index], (self.feature_dim,), name="features").copy()
                result.append((context, feature))
        return result

    def decode_batch(self, contexts, initial_noises):
        import torch
        # This numeric sampler is deliberately retained in its historical
        # module so old elevator imports and provenance remain meaningful.
        from .vlajepa_backend import flow_from_context

        if len(contexts) != len(initial_noises):
            raise ValueError("Expected one optional noise entry per context")
        if not contexts:
            return []
        noises = []
        for context, noise in zip(contexts, initial_noises):
            if noise is None:
                noises.append(context.baseline_noise)
            else:
                value = np.asarray(noise)
                if value.dtype.kind != "f":
                    raise ValueError("initial_noise must be floating point")
                value = finite_array(value, (self.horizon, self.dimension), name="initial_noise")
                noises.append(torch.as_tensor(value, device=context.tokens.device, dtype=context.tokens.dtype)[None])
        self.model.eval()
        with torch.inference_mode(), torch.autocast(self.policy.device.type, enabled=False):
            native = flow_from_context(self.model.action_model, torch.cat([item.tokens for item in contexts]),
                                       torch.cat([item.state for item in contexts]), torch.cat(noises))
        normalized = finite_array(native.float().cpu().numpy(), (len(contexts), self.horizon, self.dimension), name="native_actions")
        # Keep the historical degree/cm denormalization before SI conversion.
        actions = normalized * MODEL_SCALE[:self.dimension] * MODEL_TO_SI[:self.dimension]
        if not np.isfinite(actions).all():
            raise ValueError("Joint VLA-JEPA produced nonfinite SI actions")
        return [row.copy() for row in actions]

    def close(self):
        self.model = None
        self.policy = None


def create_backend(config):
    """Load native joint6/joint7 checkpoints with complete weights and processor."""
    allowed = {"vla_repo", "run", "checkpoint", "model_id", "device", "base_vlm", "base_encoder", "instruction_pattern"}
    if not isinstance(config, dict) or set(config) - allowed or "run" not in config:
        raise ValueError(f"vlajepa_joint requires run and supports {sorted(allowed)}")
    checked = read_checkpoint_metadata(config["run"], config.get("checkpoint"), model_id=config.get("model_id"))
    repo = activate_source("vlajepa", config.get("vla_repo"))
    helpers = importlib.import_module("starVLA.inference.piper_policy")
    if not Path(helpers.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("A different starVLA repository is already imported")
    digest = _sha256(checked["model_path"])
    if checked["expected_sha256"] is not None and checked["expected_sha256"] != digest:
        raise ValueError("Checkpoint model SHA256 mismatch")
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoProcessor
    from starVLA.model.framework.VLA_JEPA import VLA_JEPA

    device = torch.device(config.get("device", "cuda:0"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Joint VLA-JEPA loading requires an available local CUDA device")
    cfg = OmegaConf.create(checked["config"])
    for section, key in ((cfg.framework.qwenvl, "base_vlm"), (cfg.framework.vj2_model, "base_encoder")):
        path = Path(config.get(key, section[key])).expanduser().resolve(strict=True)
        if not (path / "config.json").is_file():
            raise ValueError(f"Missing local backbone configuration: {path}")
        section[key] = str(path)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cfg.framework.qwenvl.device_map = "cpu"
    model = VLA_JEPA(config=cfg)
    helpers.install_saved_processor(model, AutoProcessor.from_pretrained(checked["run"] / "processor", local_files_only=True))
    helpers.load_trained_state(model, checked["model_path"])
    helpers.disable_unused_qwen_cache(model)
    model.to(device).requires_grad_(False).eval()
    cfg.framework.qwenvl.device_map = str(device)
    from . import vlajepa_backend
    sources = {name: _sha256(repo / name) for name in (
        "starVLA/inference/piper_policy.py", "starVLA/model/framework/VLA_JEPA.py",
        "starVLA/model/modules/action_model/GR00T_ActionHeader.py", "starVLA/model/modules/vlm/QWen3.py")}
    sources.update(adapter=_sha256(__file__), sampler=_sha256(vlajepa_backend.__file__),
                   runtime_contract=_sha256(runtime_contract.__file__), image_transport=_sha256(image_transport.__file__))
    if repo == builtin_source_root("vlajepa").resolve():
        sources["bundled_model_source"] = source_tree_sha256("vlajepa", repo)
    policy = SimpleNamespace(model=model, device=device, resolution=int(cfg.datasets.vla_data.resolution_size),
        schema=checked["schema"], normalization=checked["normalization"], allowed_instructions=checked["allowed_instructions"],
        identity=dict(model_id=checked["config"]["run_id"], model_sha256=digest,
            checkpoint_step=checked["manifest"]["step"], source_revision=fingerprint(sources),
            config_sha256=_sha256(checked["run"] / "config.yaml")))
    return JointVLAJEPABackend(policy, instruction_pattern=config.get("instruction_pattern"))
