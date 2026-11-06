"""Native DDP fine-tuning of the published VLA-JEPA checkpoint on PiPER.

Run with torchrun; no Accelerate/DeepSpeed training engine is used. Coordinates
remain absolute base_link metres, followed by the first two rotation-matrix
rows. The constant gripper channel is excluded from both state and action.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


ADAPTED_KEYS = frozenset({
    "action_model.action_encoder.layer1.weight",
    "action_model.state_encoder.layer1.weight",
    "action_model.action_decoder.layer2.weight",
    "action_model.action_decoder.layer2.bias",
})
COMPONENTS = ("qwen_vl_interface.", "vj_encoder.", "vj_predictor.", "action_model.")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def training_contract(config):
    """Fingerprint every value that changes sampling, optimization or best-model selection."""
    import copy
    framework = copy.deepcopy(config["framework"])
    framework["qwenvl"].pop("device_map", None)
    framework["qwenvl"].pop("base_vlm", None)
    framework["vj2_model"].pop("base_encoder", None)
    data_keys = ("resolution_size", "per_device_batch_size", "sample_stride", "CoT_prompt")
    trainer_keys = ("max_train_steps", "num_warmup_steps", "gradient_accumulation_steps", "repeated_diffusion_steps",
                    "learning_rate", "optimizer", "weight_decay", "max_grad_norm", "enable_gradient_checkpointing",
                    "eval_interval", "val_anchors_per_episode", "val_max_samples")
    result = dict(seed=config["seed"], framework=framework,
                  data={key: config["datasets"]["vla_data"][key] for key in data_keys},
                  trainer={key: config["trainer"][key] for key in trainer_keys})
    # Preserve the original run's fingerprint when these options are absent.
    for key in ("target_epochs", "continuation_schedule"):
        if key in config["trainer"]:
            result["trainer"][key] = copy.deepcopy(config["trainer"][key])
    fingerprint = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result, fingerprint


def epoch_geometry(anchors, world, batch_size, accumulation, target_epochs=None):
    """Match DistributedSampler + DataLoader, retaining each last partial batch."""
    values = (anchors, world, batch_size, accumulation)
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("Dataset size, world size, batch and accumulation must be positive integers")
    samples_per_rank = math.ceil(anchors / world)
    batches_per_epoch = math.ceil(samples_per_rank / batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / accumulation)
    result = dict(anchors=anchors, world_size=world, per_device_batch_size=batch_size,
                  gradient_accumulation_steps=accumulation, samples_per_rank=samples_per_rank,
                  batches_per_epoch=batches_per_epoch, steps_per_epoch=steps_per_epoch,
                  sampler_padding_per_epoch=samples_per_rank * world - anchors)
    if target_epochs is not None:
        if type(target_epochs) is not int or target_epochs < 1:
            raise ValueError("target_epochs must be a positive integer")
        result.update(target_epochs=target_epochs, target_steps=target_epochs * steps_per_epoch)
    return result


def normalize_epoch_cursor(progress, batches_per_epoch):
    """Normalize at a boundary before writing checkpoint metadata."""
    if not 0 <= progress["batch_in_epoch"] <= batches_per_epoch:
        raise ValueError("Checkpoint batch cursor is outside the epoch")
    finished = progress["batch_in_epoch"] == batches_per_epoch
    if finished:
        progress["epoch"] += 1
        progress["batch_in_epoch"] = 0
    return finished


class OffsetDistributedSampler(DistributedSampler):
    """Resume the same shuffled stream without decoding discarded video batches."""
    start_index = 0

    def set_start_index(self, start_index):
        if type(start_index) is not int or not 0 <= start_index <= self.num_samples:
            raise ValueError("Invalid distributed sampler resume offset")
        self.start_index = start_index

    def __iter__(self):
        return islice(super().__iter__(), self.start_index, None)

    def __len__(self):
        return self.num_samples - self.start_index


def validate_continuation_contract(parent, current):
    """A new segment may extend its horizon/schedule, but not reinterpret data."""
    import copy
    parent, current = copy.deepcopy(parent), copy.deepcopy(current)
    for contract in (parent, current):
        for key in ("max_train_steps", "num_warmup_steps", "target_epochs", "continuation_schedule"):
            contract["trainer"].pop(key, None)
    if parent != current:
        raise ValueError("Continuation changed model, data, batch, seed, optimizer or validation semantics")


def continuation_schedule(training, learning_rates, maximum, warmup):
    """Restart LR smoothly from the stored rate without resetting AdamW moments."""
    start = int(training["step"])
    if maximum <= start or warmup < 0 or warmup >= maximum - start:
        raise ValueError("Continuation must extend the checkpoint with room for warmup and cosine decay")
    factors = []
    names = ["qwen_vl_interface", "vj_predictor", "action_model"]
    groups = training["optimizer"]["param_groups"]
    if [group.get("name") for group in groups] != names:
        raise ValueError("Continuation optimizer parameter-group ordering differs")
    for group in groups:
        base = float(learning_rates[group["name"]])
        if base <= 0 or group.get("initial_lr") != base:
            raise ValueError("Continuation optimizer base learning rate differs")
        factor = float(group["lr"]) / base
        if not math.isfinite(factor) or not 0 <= factor <= 1:
            raise ValueError("Unsupported checkpoint learning-rate factor")
        factors.append(factor)
    if training["scheduler"]["last_epoch"] != start:
        raise ValueError("Checkpoint scheduler cursor does not match optimizer-step cursor")
    return dict(kind="checkpoint_lr_linear_warmup_cosine", start_step=start,
                warmup_steps=warmup, end_step=maximum, start_factors=factors, end_factor=.1)


def lr_functions(maximum, warmup, continuation=None):
    if maximum < 1 or warmup < 0:
        raise ValueError("max_train_steps must be positive and warmup nonnegative")
    if continuation is None:
        def factor(step):
            if step < warmup:
                return (step + 1) / max(warmup, 1)
            fraction = min(1., (step - warmup) / max(1, maximum - warmup))
            return .1 + .9 * .5 * (1 + math.cos(math.pi * fraction))
        return factor
    if (continuation["kind"] != "checkpoint_lr_linear_warmup_cosine"
            or continuation["end_step"] != maximum or continuation["warmup_steps"] != warmup):
        raise ValueError("Continuation schedule disagrees with training horizon")
    def for_group(initial):
        def factor(step):
            offset = max(0, step - continuation["start_step"])
            if offset <= warmup and warmup:
                return initial + (1. - initial) * offset / warmup
            duration = maximum - continuation["start_step"] - warmup
            fraction = min(1., max(0., (offset - warmup) / duration))
            return continuation["end_factor"] + (1. - continuation["end_factor"]) * .5 * (1 + math.cos(math.pi * fraction))
        return factor
    return [for_group(value) for value in continuation["start_factors"]]


def select_pretrained_state(source, target):
    """Fail closed except for the named 9D interfaces.

    ``strict=False`` alone does not permit shape mismatches. Every skipped
    tensor is listed, and backbone/predictor absence is always an error.
    """
    if not isinstance(source, dict) or not source:
        raise ValueError("Pretrained checkpoint must contain a nonempty tensor state_dict")
    if all(key.startswith("module.") for key in source):
        source = {key[len("module."):]: value for key, value in source.items()}
    if any(not isinstance(value, torch.Tensor) for value in source.values()):
        raise ValueError("Pretrained state_dict contains non-tensors")
    absent_head = not any(key.startswith("action_model.") for key in source)
    if absent_head:
        raise ValueError("Entire pretrained action head is absent; the verified official checkpoint has 248 action_model tensors")
    unexpected = sorted(set(source) - set(target))
    if unexpected:
        raise ValueError(f"Unexpected pretrained keys: {unexpected}")
    missing = sorted(set(target) - set(source))
    if missing:
        raise ValueError(f"Missing pretrained backbone/interface keys: {missing}")
    loaded, adapted = {}, []
    for key, value in source.items():
        if tuple(value.shape) == tuple(target[key].shape):
            loaded[key] = value
        elif key in ADAPTED_KEYS:
            expected = list(target[key].shape)
            dimension = 1 if key.endswith("layer1.weight") else 0
            if expected[dimension] != 9:
                raise ValueError(f"Expected a 9D target interface for {key}")
            expected[dimension] = 8 if ".state_encoder." in key else 7
            if list(value.shape) != expected:
                raise ValueError(f"Not the official 7D-action/8D-state adaptation at {key}: {tuple(value.shape)}")
            adapted.append(dict(key=key, source_shape=list(value.shape), target_shape=list(target[key].shape)))
        else:
            raise ValueError(f"Unapproved pretrained shape mismatch {key}: {tuple(value.shape)} != {tuple(target[key].shape)}")
    for prefix in COMPONENTS[:3]:
        if not any(key.startswith(prefix) for key in loaded):
            raise ValueError(f"No pretrained tensors loaded for {prefix}")
    report = dict(head_initialization="pretrained_head_with_new_9d_interfaces",
                  missing_keys=missing, adapted_keys=adapted, unexpected_keys=[],
                  loaded_keys=len(loaded), loaded_numel=sum(v.numel() for v in loaded.values()),
                  components={prefix: dict(tensors=sum(k.startswith(prefix) for k in loaded),
                                          numel=sum(v.numel() for k, v in loaded.items() if k.startswith(prefix)))
                              for prefix in COMPONENTS})
    return loaded, report


def validate_tied_source(model, source):
    """A shared target parameter must not silently receive conflicting tensors."""
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    checked = []
    for names in aliases.values():
        if len(names) < 2:
            continue
        present = [name for name in names if name in source]
        if not present:
            continue
        if len(present) != len(names):
            raise ValueError(f"Only some aliases of a tied target parameter were provided: {names}")
        first = source[names[0]]
        for name in names[1:]:
            if not torch.equal(first, source[name]):
                raise ValueError(f"Conflicting source tensors for tied target parameter: {names[0]} != {name}")
        checked.append(names)
    return checked


def load_pretrained(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    selected, report = select_pretrained_state(checkpoint, model.state_dict())
    report["tied_parameter_groups_checked"] = validate_tied_source(model, selected)
    result = model.load_state_dict(selected, strict=False)
    expected_missing = set(report["missing_keys"]) | {entry["key"] for entry in report["adapted_keys"]}
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError("Loaded checkpoint differs from the verified key plan")
    return report


def configure_parameters(model, checkpointing):
    """Freeze fixed teachers/unused branches, retaining trainable tied inputs."""
    model.requires_grad_(True)
    frozen = [model.vj_encoder, model.vj_predictor.state_encoder, model.vj_predictor.extrinsics_encoder]
    qwen = model.qwen_vl_interface.model
    visual = getattr(qwen, "visual", None)
    if visual is None:
        visual = qwen.model.visual
    frozen.append(visual)
    embedding_ids = {id(p) for p in qwen.get_input_embeddings().parameters()}
    lm_head = qwen.get_output_embeddings()
    untied_head = all(id(p) not in embedding_ids for p in lm_head.parameters())
    if untied_head:
        frozen.append(lm_head)
    for module in frozen:
        module.requires_grad_(False)
        module.eval()
    qwen.config.use_cache = False
    if hasattr(qwen.config, "text_config"):
        qwen.config.text_config.use_cache = False
    if checkpointing:
        qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.vj_predictor.use_activation_checkpointing = True
    # FP32 trainable weights give AdamW FP32 master weights and moments. Actual
    # transformer matmuls use BF16 autocast; the action expert stays FP32.
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    report = dict(lm_head_tied=not untied_head, gradient_checkpointing=bool(checkpointing),
                  trainable_numel=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  frozen_numel=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                  groups={prefix: dict(trainable_numel=sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix) and p.requires_grad),
                                       frozen_numel=sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix) and not p.requires_grad))
                          for prefix in COMPONENTS})
    if not report["groups"]["qwen_vl_interface."]["trainable_numel"]:
        raise RuntimeError("Real pretrained language-model fine-tuning requires trainable Qwen parameters")
    return frozen, report


def validation_anchors(dataset, anchors_per_episode, max_samples=0):
    """Evenly spaced anchors in each held-out episode, including both ends."""
    anchors = []
    for source_id in dataset.source_episode_ids:
        length = dataset.episode_lengths[source_id]
        count = length if anchors_per_episode == 0 else min(length, anchors_per_episode)
        anchors.extend((int(source_id), int(frame)) for frame in np.unique(np.linspace(0, length - 1, count).round().astype(int)))
    if max_samples and max_samples < len(anchors):
        indices = np.linspace(0, len(anchors) - 1, max_samples).round().astype(int)
        anchors = [anchors[index] for index in indices]
    return anchors


def rotation_errors_deg(prediction, target):
    """Project row-major rot6 to SO(3); degenerate estimates receive 180 deg."""
    def project(value):
        first, second = value[..., :3], value[..., 3:]
        first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
        row0 = first / np.maximum(first_norm, 1e-12)
        row1 = second - (row0 * second).sum(-1, keepdims=True) * row0
        second_norm = np.linalg.norm(row1, axis=-1, keepdims=True)
        row1 /= np.maximum(second_norm, 1e-12)
        valid = (first_norm[..., 0] > 1e-10) & (second_norm[..., 0] > 1e-10)
        return np.stack((row0, row1, np.cross(row0, row1)), axis=-2), valid
    predicted, valid = project(np.asarray(prediction, dtype=np.float64))
    truth, truth_valid = project(np.asarray(target, dtype=np.float64))
    if not truth_valid.all():
        raise ValueError("Ground-truth rotation is degenerate")
    cosine = ((predicted * truth).sum(axis=(-1, -2)) - 1) / 2
    error = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return np.where(valid, error, 180.), ~valid


METRIC_FIELDS = ("anchors", "valid_actions", "position_sum_m", "rotation_sum_deg", "degenerate_rotations",
                 "first_position_sum_m", "first_rotation_sum_deg", "baseline_position_sum_m", "baseline_rotation_sum_deg",
                 "flow_sum", "flow_count", "wm_sum", "wm_count")


@torch.no_grad()
def evaluate(model, dataset, anchors, rank, world, device, seed, step, output):
    """Actual stochastic action generation, using reproducible per-anchor noise."""
    model.eval()
    totals = np.zeros((13, len(METRIC_FIELDS)), dtype=np.float64)
    started = time.monotonic()
    rank0_total = len(range(0, len(anchors), world))
    if rank == 0:
        print(json.dumps(dict(event="validation_started", step=step, global_anchors=len(anchors),
                              rank0_anchors=rank0_total)), flush=True)
    for ordinal in range(rank, len(anchors), world):
        source_id, frame = anchors[ordinal]
        example = dataset.get_frame(source_id, frame)
        # This context restores training RNG after every validation example.
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(seed + source_id * 100003 + frame)
            torch.cuda.manual_seed(seed + source_id * 100003 + frame)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model.predict_action([example["image"]], [example["lang"]], state=np.stack([example["state"]]))["normalized_actions"][0]
                losses = model([example])
        target = example["action"]
        if prediction.shape != target.shape or not np.isfinite(prediction).all():
            raise RuntimeError(f"Invalid generated actions at source={source_id} frame={frame}")
        valid = ~example["action_is_pad"]
        position = np.linalg.norm(prediction[:, :3] - target[:, :3], axis=-1)
        orientation, degenerate = rotation_errors_deg(prediction[:, 3:], target[:, 3:])
        state = example["state"][0]
        baseline_pos = np.linalg.norm(state[:3] - target[0, :3])
        baseline_rot = rotation_errors_deg(state[None, 3:], target[:1, 3:])[0][0]
        future_valid = (~example["video_is_pad"].reshape(-1, model.vj_encoder.config.tubelet_size).any(-1))[1:].sum()
        values = np.array([1, valid.sum(), position[valid].sum(), orientation[valid].sum(), degenerate[valid].sum(),
                           position[0], orientation[0], baseline_pos, baseline_rot,
                           float(losses["action_loss"]) * valid.sum(), valid.sum(),
                           float(losses["wm_loss"]) * future_valid, future_valid], dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError("Nonfinite validation loss/metric")
        totals[0] += values
        totals[int(example["floor"]) - 23] += values
        local_completed = ordinal // world + 1
        if rank == 0 and local_completed % 25 == 0:
            print(json.dumps(dict(event="validation_progress", step=step, rank0_completed=local_completed,
                                  rank0_anchors=rank0_total, seconds=time.monotonic() - started)), flush=True)
    tensor = torch.as_tensor(totals, device=device)
    dist.all_reduce(tensor)
    totals = tensor.cpu().numpy()
    def summarize(row):
        n, action_n = max(row[0], 1), max(row[1], 1)
        return dict(anchors=int(row[0]), valid_actions=int(row[1]),
                    position_mean_m=float(row[2] / action_n), rotation_mean_deg=float(row[3] / action_n),
                    degenerate_rotations=int(row[4]), first_action_position_mean_m=float(row[5] / n),
                    first_action_rotation_mean_deg=float(row[6] / n),
                    state_copy_baseline_first_position_mean_m=float(row[7] / n),
                    state_copy_baseline_first_rotation_mean_deg=float(row[8] / n),
                    flow_loss=float(row[9] / max(row[10], 1)),
                    wm_loss_weighted_0_1=float(row[11] / max(row[12], 1)))
    result = dict(step=step, evaluation_seed=seed, seconds=time.monotonic() - started,
                  overall=summarize(totals[0]), floors={str(floor): summarize(totals[floor - 23]) for floor in range(24, 36)})
    if result["overall"]["anchors"] != len(anchors):
        raise RuntimeError("Distributed validation skipped or duplicated anchors")
    if rank == 0:
        write_json(output / f"validation_{step:06d}.json", result)
        print(json.dumps(dict(event="validation", **result["overall"], step=step)), flush=True)
    return result


def loss_weights(batches, tubelet_size, device, world):
    local = torch.tensor([[sum((~example["action_is_pad"]).sum() for example in batch),
                           sum((~example["video_is_pad"].reshape(-1, tubelet_size).any(-1))[1:].sum() for example in batch)]
                          for batch in batches], device=device, dtype=torch.float64)
    total = local.sum(0)
    dist.all_reduce(total)
    return (world * local / total.clamp_min(1)).float()


def get_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def parameter_probes(model):
    """Small deterministic weight slices, including a pretrained language layer."""
    probes = {}
    choices = (("qwen_vl_interface.", "q_proj.weight"),
               ("vj_predictor.", "predictor_embed.weight"),
               ("action_model.", "action_decoder.layer2.weight"))
    for prefix, suffix in choices:
        candidates = [(name, p) for name, p in model.named_parameters()
                      if p.requires_grad and name.startswith(prefix) and name.endswith(suffix)]
        if not candidates:
            raise RuntimeError(f"Cannot identify an actual trainable parameter probe for {prefix}")
        name, parameter = candidates[0]
        probes[name] = parameter.detach().flatten()[:4096].float().cpu().clone()
    return probes


def save_checkpoint(model, optimizer, scheduler, output, progress, is_best, rank, world, provenance):
    states = [None] * world
    dist.all_gather_object(states, get_rng())
    if rank == 0:
        root = output / "checkpoints"
        root.mkdir(exist_ok=True)
        name = f"step_{progress['step']:06d}"
        destination = root / name
        staging = root / (name + ".tmp")
        staging.mkdir(exist_ok=False)
        # Plain upstream-compatible keys; no wrapper/module prefix.
        torch.save(model.state_dict(), staging / "model.pt")
        torch.save(dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        rng_by_rank=states, world_size=world, **progress,
                        **provenance),
                   staging / "training.pt")
        write_json(staging / "manifest.json", dict(complete=True, **progress,
                   **provenance,
                   model_sha256=sha256(staging / "model.pt"), training_sha256=sha256(staging / "training.pt")))
        os.replace(staging, destination)
        for alias in ("last", "best") if is_best else ("last",):
            temporary = root / (alias + ".tmp")
            temporary.symlink_to(name, target_is_directory=True)
            os.replace(temporary, root / alias)
        keep = {(root / alias).resolve() for alias in ("best", "last") if (root / alias).exists()}
        for path in root.glob("step_*"):
            if path.is_dir() and path.resolve() not in keep and not path.name.endswith(".tmp"):
                shutil.rmtree(path)
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="scripts/configs/vlajepa_piper_ft.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--target-epochs", type=int, help="Cumulative full epochs, including the source checkpoint's progress")
    parser.add_argument("--val-max-samples", type=int)
    restarts = parser.add_mutually_exclusive_group()
    restarts.add_argument("--resume", type=Path, help="Resume an unchanged run and schedule")
    restarts.add_argument("--continue-from", type=Path, help="Extend a complete checkpoint into a new run, retaining optimizer/RNG")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    from omegaconf import OmegaConf
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.set))
    if args.run_id:
        cfg.run_id = args.run_id
    if args.max_steps is not None:
        cfg.trainer.max_train_steps = args.max_steps
    if args.target_epochs is not None:
        cfg.trainer.target_epochs = args.target_epochs
    if args.max_steps is not None and cfg.trainer.get("target_epochs") is not None:
        raise ValueError("Use either --max-steps or target_epochs, not both")
    if args.val_max_samples is not None:
        cfg.trainer.val_max_samples = args.val_max_samples
    if cfg.framework.action_model.action_dim != 9 or cfg.framework.action_model.state_dim != 9:
        raise ValueError("PiPER state/action must be raw xyz + row-major rot6, with no gripper")
    rank, world, local_rank = (int(os.environ.get(key, default)) for key, default in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=60))
    torch.cuda.set_per_process_memory_fraction(min(1., float(cfg.trainer.memory_limit_gib) * 1024**3 / torch.cuda.get_device_properties(device).total_memory), device)
    cfg.framework.qwenvl.device_map = f"cuda:{local_rank}"
    seed_all(int(cfg.seed))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output = Path(cfg.run_root_dir) / cfg.run_id
    if rank == 0:
        if output.exists() and any(output.iterdir()) and args.resume is None:
            raise FileExistsError(f"Run directory is not empty: {output}; select a new --run-id or --resume")
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    from starVLA.dataloader.piper_lerobot import PiperLeRobotDataset, collate_fn
    common = dict(root=cfg.datasets.vla_data.data_root_dir, split_manifest=output / "split.json",
                  action_horizon=7, video_horizon=8, resolution=int(cfg.datasets.vla_data.resolution_size), seed=int(cfg.seed),
                  frame_cache_size=int(cfg.datasets.vla_data.frame_cache_size), decoder_cache_size=int(cfg.datasets.vla_data.decoder_cache_size))
    train_data = PiperLeRobotDataset(split="train", sample_stride=int(cfg.datasets.vla_data.sample_stride), **common)
    val_data = PiperLeRobotDataset(split="val", **common)
    if rank == 0:
        print(json.dumps(dict(event="dataset_ready", train_episodes=len(train_data.source_episode_ids),
                              validation_episodes=len(val_data.source_episode_ids), train_anchors=len(train_data),
                              dataset_sha256=train_data.export_manifest_sha256)), flush=True)
    anchors = validation_anchors(val_data, int(cfg.trainer.val_anchors_per_episode), int(cfg.trainer.val_max_samples))
    if not anchors:
        raise ValueError("Validation must include at least one held-out anchor")
    provenance = dict(dataset_sha256=train_data.export_manifest_sha256, split_sha256=train_data.split_manifest_sha256)
    accumulation = int(cfg.trainer.gradient_accumulation_steps)
    batch_size = int(cfg.datasets.vla_data.per_device_batch_size)
    geometry = epoch_geometry(len(train_data), world, batch_size, accumulation, cfg.trainer.get("target_epochs"))
    if "target_steps" in geometry:
        cfg.trainer.max_train_steps = geometry["target_steps"]
    maximum, warmup = int(cfg.trainer.max_train_steps), int(cfg.trainer.num_warmup_steps)
    restart = args.resume or args.continue_from
    progress = dict(step=0, epoch=0, batch_in_epoch=0, best_score=None)
    lineage = None
    if restart:
        checkpoint_dir = restart.resolve(strict=True)
        if checkpoint_dir.parent.name != "checkpoints":
            raise ValueError("Restart requires run/checkpoints/{best,last,step_*}")
        if args.continue_from and output.resolve() == checkpoint_dir.parent.parent:
            raise ValueError("Continuation must write to a new run; the parent checkpoint is immutable")
        metadata = json.loads((checkpoint_dir / "manifest.json").read_text())
        if metadata.get("complete") is not True:
            raise ValueError("Restart checkpoint is not complete")
        for name in ("model", "training"):
            if sha256(checkpoint_dir / (name + ".pt")) != metadata[name + "_sha256"]:
                raise ValueError(f"Restart {name} checksum mismatch")
        training = torch.load(checkpoint_dir / "training.pt", map_location="cpu", weights_only=False, mmap=True)
        if training["world_size"] != world or any(training.get(key) != value for key, value in provenance.items()):
            raise ValueError("Restart world size, dataset or episode split differs")
        for key in (*progress, *provenance, "training_fingerprint"):
            if training.get(key) != metadata.get(key):
                raise ValueError(f"Restart manifest/training state mismatch: {key}")
        progress = {key: training[key] for key in progress}
        normalize_epoch_cursor(progress, geometry["batches_per_epoch"])
        expected_step = progress["epoch"] * geometry["steps_per_epoch"] + math.ceil(progress["batch_in_epoch"] / accumulation)
        if (progress["step"] != expected_step
                or progress["batch_in_epoch"] % accumulation != 0):
            raise ValueError("Restart progress does not match batch/accumulation geometry")
        if maximum <= progress["step"]:
            raise ValueError("Training target must exceed the checkpoint's completed steps")
        if args.continue_from:
            cfg.trainer.continuation_schedule = continuation_schedule(training, cfg.trainer.learning_rate, maximum, warmup)
    elif cfg.trainer.get("continuation_schedule") is not None:
        raise ValueError("A continuation schedule requires --resume or --continue-from")
    contract, fingerprint = training_contract(OmegaConf.to_container(cfg, resolve=True))
    provenance["training_fingerprint"] = fingerprint
    if args.continue_from:
        source_contract = json.loads((checkpoint_dir.parent.parent / "training_contract.json").read_text())
        source_fingerprint = hashlib.sha256(json.dumps(source_contract["contract"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if source_contract["fingerprint"] != source_fingerprint or training["training_fingerprint"] != source_fingerprint:
            raise ValueError("Parent training contract fingerprint mismatch")
        validate_continuation_contract(source_contract["contract"], contract)
        lineage = dict(parent_checkpoint=str(checkpoint_dir), parent_model_sha256=metadata["model_sha256"],
                       parent_training_sha256=metadata["training_sha256"], parent_manifest_sha256=sha256(checkpoint_dir / "manifest.json"),
                       parent_training_fingerprint=source_fingerprint, parent_training_contract=source_contract["contract"],
                       parent_progress=dict(progress), parent_world_size=world, **geometry,
                       restored=["all_model_weights", "AdamW_parameter_groups_and_moments", "rank_RNG", "sampler_cursor"],
                       learning_rate_schedule=OmegaConf.to_container(cfg.trainer.continuation_schedule, resolve=True),
                       best_selection="best validation score within continuation segment; parent kept in its original run")
        progress["best_score"] = None
    elif args.resume:
        if training.get("training_fingerprint") != fingerprint:
            raise ValueError("Resume training contract differs; use --continue-from for an explicit new schedule segment")
        lineage_path = checkpoint_dir.parent.parent / "continuation.json"
        if lineage_path.exists():
            lineage = json.loads(lineage_path.read_text())
    from starVLA.model.framework.VLA_JEPA import VLA_JEPA
    model = VLA_JEPA(config=cfg)
    if restart:
        from transformers import AutoProcessor
        from starVLA.inference.piper_policy import install_saved_processor
        processor_dir = checkpoint_dir.parent.parent / "processor"
        processor_hashes = {str(path.relative_to(processor_dir)): sha256(path)
                            for path in sorted(processor_dir.rglob("*")) if path.is_file()}
        if not processor_hashes:
            raise ValueError("Restart requires the source run's saved processor")
        install_saved_processor(model, AutoProcessor.from_pretrained(processor_dir, local_files_only=True))
        if lineage is not None and args.continue_from:
            lineage["parent_processor_sha256"] = processor_hashes
    if rank == 0:
        print(json.dumps(dict(event="model_constructed", parameter_numel=sum(p.numel() for p in model.parameters()))), flush=True)
    loading = None if restart else load_pretrained(model, cfg.trainer.pretrained_checkpoint)
    if rank == 0 and loading is not None:
        print(json.dumps(dict(event="pretrained_loaded", loaded_keys=loading["loaded_keys"], loaded_numel=loading["loaded_numel"],
                              adapted_keys=loading["adapted_keys"], tied_parameter_groups_checked=loading["tied_parameter_groups_checked"])), flush=True)
    model.to(device)
    frozen, parameter_report = configure_parameters(model, cfg.trainer.enable_gradient_checkpointing)
    learning_rates = cfg.trainer.learning_rate
    groups = [dict(params=[p for name, p in model.named_parameters() if name.startswith(prefix) and p.requires_grad],
                   lr=float(learning_rates[prefix.rstrip(".")] ), name=prefix.rstrip("."))
              for prefix in ("qwen_vl_interface.", "vj_predictor.", "action_model.")]
    if sum(sum(p.numel() for p in group["params"]) for group in groups) != parameter_report["trainable_numel"]:
        raise RuntimeError("Optimizer groups omit or duplicate trainable parameters")
    optimizer = torch.optim.AdamW(groups, betas=tuple(cfg.trainer.optimizer.betas), eps=float(cfg.trainer.optimizer.eps),
                                 weight_decay=float(cfg.trainer.weight_decay), foreach=False)
    if restart:
        resumed_weights = torch.load(checkpoint_dir / "model.pt", map_location="cpu", weights_only=True, mmap=True)
        validate_tied_source(model, resumed_weights)
        # configure_parameters already promotes trainable tensors to FP32. Also
        # preserve any saved frozen-tensor dtype rather than silently rounding.
        for name, tensor in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False)):
            if name in resumed_weights and tensor.dtype != resumed_weights[name].dtype:
                tensor.data = tensor.data.to(dtype=resumed_weights[name].dtype)
        model.load_state_dict(resumed_weights, strict=True)
        loading = dict(initialization="full_trained_checkpoint", loaded_keys=len(resumed_weights),
                       checkpoint_path=str(checkpoint_dir), checkpoint_sha256=metadata["model_sha256"],
                       optimizer_restored=True, source_step=progress["step"],
                       saved_processor_files_sha256=processor_hashes)
        del resumed_weights
        optimizer.load_state_dict(training["optimizer"])
        optimizer_steps = [int(state["step"]) for state in optimizer.state.values() if "step" in state]
        if not optimizer_steps or any(value != progress["step"] for value in optimizer_steps):
            raise ValueError("Restored AdamW moments have inconsistent optimizer-step counters")
        loading["optimizer_state_evidence"] = dict(parameter_states=len(optimizer_steps),
                                                  min_step=min(optimizer_steps), max_step=max(optimizer_steps))
        resume_rng = training["rng_by_rank"][rank]
    schedule = OmegaConf.to_container(cfg.trainer.continuation_schedule, resolve=True) if cfg.trainer.get("continuation_schedule") else None
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_functions(maximum, warmup, schedule),
                 last_epoch=progress["step"] - 1 if args.continue_from else -1)
    if args.resume:
        scheduler.load_state_dict(training["scheduler"])
        # LambdaLR construction evaluates step zero; restore the saved rates too.
        for group, rate in zip(optimizer.param_groups, scheduler.get_last_lr()):
            group["lr"] = rate
    if restart:
        del training
    initial_probes = parameter_probes(model)
    if rank == 0:
        if not restart:
            loading["checkpoint_path"] = str(Path(cfg.trainer.pretrained_checkpoint).resolve())
            loading["checkpoint_sha256"] = sha256(cfg.trainer.pretrained_checkpoint)
        write_json(output / "pretrained_loading.json", loading)
        write_json(output / "epoch_geometry.json", geometry)
        if lineage is not None:
            write_json(output / "continuation.json", lineage)
        write_json(output / "parameters.json", parameter_report)
        write_json(output / "validation_anchors.json", dict(anchors=anchors, **provenance))
        write_json(output / "training_contract.json", dict(fingerprint=fingerprint, contract=contract))
        write_json(output / "action_schema.json", dict(state_dim=9, action_dim=9, action_horizon=7, video_horizon=8,
                   fps=30, pose_frame="base_link", pose_link="gripper_tcp", xyz_units="metres", normalization="none",
                   rotation="R[:2,:].reshape(6); first two rows", gripper="excluded; fixed by controller",
                   episode_end="first sampled target light on", terminal_action="repeat previous planned target", **provenance))
        saved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        saved_cfg.framework.qwenvl.device_map = "cuda"
        if not (output / "config.yaml").exists():
            OmegaConf.save(saved_cfg, output / "config.yaml")
        model.qwen_vl_interface.processor.save_pretrained(output / "processor")
        print(json.dumps(dict(event="initialized", **parameter_report, pretrained=loading)), flush=True)
    ddp = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False, broadcast_buffers=False, gradient_as_bucket_view=True)
    if rank == 0:
        print(json.dumps(dict(event="ddp_ready", world_size=world, trainable_numel=parameter_report["trainable_numel"])), flush=True)
    sampler = OffsetDistributedSampler(train_data, num_replicas=world, rank=rank, shuffle=True, seed=int(cfg.seed), drop_last=False)
    generator = torch.Generator().manual_seed(int(cfg.seed) + rank * 1009)
    num_workers = int(cfg.datasets.vla_data.num_workers)
    loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler,
                        num_workers=num_workers, collate_fn=collate_fn, drop_last=False, pin_memory=False,
                        persistent_workers=num_workers > 0, worker_init_fn=worker_seed, generator=generator,
                        **({"multiprocessing_context": "spawn", "prefetch_factor": 2} if num_workers else {}))
    seed_all(int(cfg.seed) + rank * 1009)
    if restart:
        restore_rng(resume_rng)
    else:
        evaluate(model, val_data, anchors, rank, world, device, int(cfg.seed) + 500000, 0, output)
    epoch_batches = geometry["batches_per_epoch"]
    if rank == 0:
        ready = dict(event="training_ready", **progress, **geometry,
                     resume_sampler_offset=progress["batch_in_epoch"] * batch_size,
                     global_batch_size=world * batch_size * accumulation,
                     learning_rates=scheduler.get_last_lr(), continuation=lineage is not None)
        ready["target_steps"] = maximum
        write_json(output / "initialization.json", ready)
        print(json.dumps(ready), flush=True)
    optimizer.zero_grad(set_to_none=True)
    start = time.monotonic()
    checked_gradients = False
    while progress["step"] < maximum:
        sampler.set_epoch(progress["epoch"])
        sampler.set_start_index(progress["batch_in_epoch"] * batch_size)
        model.train()
        for module in frozen:
            module.eval()
        iterator = iter(loader)
        while progress["batch_in_epoch"] < epoch_batches and progress["step"] < maximum:
            microbatches = min(accumulation, epoch_batches - progress["batch_in_epoch"])
            batches = [next(iterator) for _ in range(microbatches)]
            weights = loss_weights(batches, model.vj_encoder.config.tubelet_size, device, world)
            logged = torch.zeros(2, device=device)
            for micro, batch in enumerate(batches):
                context = ddp.no_sync() if micro + 1 < microbatches else nullcontext()
                with context:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        losses = ddp(batch)
                        components = torch.stack((losses["action_loss"], losses["wm_loss"])) * weights[micro]
                        loss = components.sum()
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    loss.backward()
                logged += components.detach()
                progress["batch_in_epoch"] += 1
            if not checked_gradients:
                missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
                if missing:
                    raise RuntimeError(f"Trainable parameters outside the actual loss graph: {missing}")
                checked_gradients = True
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(cfg.trainer.max_grad_norm), error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            progress["step"] += 1
            epoch_finished = normalize_epoch_cursor(progress, epoch_batches)
            dist.all_reduce(logged)
            if rank == 0 and (progress["step"] == 1 or progress["step"] % int(cfg.trainer.logging_frequency) == 0):
                record = dict(event="train", step=progress["step"], epoch=progress["epoch"],
                              action_loss=float(logged[0] / world), wm_loss_weighted_0_1=float(logged[1] / world),
                              grad_norm=float(norm), learning_rates=scheduler.get_last_lr(), seconds=time.monotonic()-start,
                              peak_memory_gib=torch.cuda.max_memory_allocated(device) / 1024**3)
                with (output / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            if progress["step"] % int(cfg.trainer.eval_interval) == 0 or progress["step"] == maximum:
                metrics = evaluate(model, val_data, anchors, rank, world, device, int(cfg.seed)+500000, progress["step"], output)
                # Fixed explicit score; physical metrics remain separate in reports.
                score = metrics["overall"]["first_action_position_mean_m"] + .1 * math.radians(metrics["overall"]["first_action_rotation_mean_deg"])
                is_best = progress["best_score"] is None or score < progress["best_score"]
                if is_best:
                    progress["best_score"] = score
                save_checkpoint(model, optimizer, scheduler, output, progress, is_best, rank, world, provenance)
                model.train()
                for module in frozen:
                    module.eval()
            if epoch_finished:
                break
    if rank == 0:
        parameters = dict(model.named_parameters())
        changes = {}
        for name, before in initial_probes.items():
            after = parameters[name].detach().flatten()[:len(before)].float().cpu()
            difference = (after - before).abs()
            changes[name] = dict(elements=len(before), changed_elements=int((difference != 0).sum()),
                                 max_abs_change=float(difference.max()), mean_abs_change=float(difference.mean()),
                                 initial_sha256=hashlib.sha256(before.numpy().tobytes()).hexdigest(),
                                 final_sha256=hashlib.sha256(after.numpy().tobytes()).hexdigest())
        write_json(output / "parameter_update_evidence.json", dict(baseline="resumed checkpoint" if restart else "loaded official pretrained plus new interfaces",
                   final_step=progress["step"], probes=changes))
        write_json(output / "completed.json", dict(success=True, **progress, **provenance,
                   seconds=time.monotonic()-start, world_size=world,
                   best_checkpoint=str((output/"checkpoints/best").resolve()), last_checkpoint=str((output/"checkpoints/last").resolve())))
    train_data.close()
    val_data.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
