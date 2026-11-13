"""Fine-tune LeRobot π0.5 and export its native inference checkpoint."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

from pressb.training.image_transforms import resolve_image_transforms_config


def policy_config(config):
    import draccus
    from pressb.models.pi05 import load_pi05_config, parse_pi05_config

    options = dict(config)
    if options.pop("type", "pi05") != "pi05":
        raise ValueError("Expected policy.type=pi05")
    checkpoint = options.pop("pretrained_path", None)
    defaults = {}
    if checkpoint is not None:
        checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
        loaded = load_pi05_config(checkpoint, device=options.get("device"))
        defaults = draccus.encode(loaded)
        defaults.pop("type", None)
    options = {**defaults, **options, "push_to_hub": False, "pretrained_path": None,
               "input_features": {}, "output_features": {}}
    policy = parse_pi05_config(options)
    if getattr(policy, "use_peft", False) or policy.rtc_config is not None:
        raise ValueError("PressB π0.5 SFT supports native absolute-action weight training")
    if options.get("device") and str(options["device"]).split(":")[0] != policy.device.split(":")[0]:
        raise ValueError(f"π0.5 training device {options['device']!r} is unavailable")
    return policy, checkpoint


def set_features(policy, dataset, wire):
    from lerobot.datasets.utils import dataset_to_policy_features
    from pressb.training.pi05_data import ACTION, STATE

    source = dataset.source.meta
    inputs = [STATE, *wire["camera_keys"].values()]
    features = dataset_to_policy_features({key: source.features[key] for key in [*inputs, ACTION]})
    if (features[STATE].shape[0] > policy.max_state_dim
            or features[ACTION].shape[0] > policy.max_action_dim):
        raise ValueError("π0.5 state/action dimensions exceed the configured model capacity")
    if wire["action"].get("horizon", policy.chunk_size) != policy.chunk_size:
        raise ValueError("pressb.action.horizon must match policy.chunk_size")
    policy.input_features = {key: features[key] for key in inputs}
    policy.output_features = {ACTION: features[ACTION]}


def make_processors(policy, stats, tokenizer_path):
    from lerobot.policies.pi05.processor_pi05 import (
        Pi05PrepareStateTokenizerProcessorStep,
        make_pi05_pre_post_processors,
    )
    from lerobot.processor import (
        AddBatchDimensionProcessorStep,
        DeviceProcessorStep,
        NormalizerProcessorStep,
        PolicyProcessorPipeline,
        RenameObservationsProcessorStep,
        TokenizerProcessorStep,
        UnnormalizerProcessorStep,
    )
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    if tokenizer_path is None:
        return make_pi05_pre_post_processors(policy, stats)
    return (
        PolicyProcessorPipeline(
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            steps=[
                RenameObservationsProcessorStep(rename_map={}),
                AddBatchDimensionProcessorStep(),
                NormalizerProcessorStep(features={**policy.input_features, **policy.output_features},
                                        norm_map=policy.normalization_mapping, stats=stats),
                Pi05PrepareStateTokenizerProcessorStep(max_state_dim=policy.max_state_dim),
                TokenizerProcessorStep(tokenizer_name=str(tokenizer_path), max_length=policy.tokenizer_max_length,
                                       padding_side="right", padding="max_length"),
                DeviceProcessorStep(device=policy.device),
            ],
        ),
        PolicyProcessorPipeline(
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            steps=[
                UnnormalizerProcessorStep(features=policy.output_features, norm_map=policy.normalization_mapping,
                                          stats=stats),
                DeviceProcessorStep(device="cpu"),
            ],
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def _batch(batch, cameras, preprocessor):
    import torch

    for camera in cameras:
        if batch[camera].dtype == torch.uint8:
            batch[camera] = batch[camera].float().div_(255.)
    return preprocessor(batch)


def validate(model, loader, preprocessor, cameras, device, seed, *, bf16):
    import torch

    model.eval()
    devices = [device.index] if device.type in ("cuda", "npu") else []
    rng_kwargs = {"device_type": device.type} if devices else {}
    total, count = 0., 0
    with (torch.no_grad(), torch.random.fork_rng(devices=devices, **rng_kwargs),
          torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16)):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            module = getattr(torch, device.type)
            with module.device(device):
                module.manual_seed(seed)
        for raw in loader:
            batch = _batch(raw, cameras, preprocessor)
            loss, _ = model(batch)
            n = batch["action"].shape[0]
            total += float(loss) * n
            count += n
    score = total / count
    if not math.isfinite(score):
        raise FloatingPointError("Nonfinite π0.5 validation loss")
    return score


def _save_checkpoint(output, step, model, preprocessor, postprocessor, wire, *, best, keep):
    folder = output / "checkpoints"
    folder.mkdir(exist_ok=True)
    checkpoint = folder / f"step_{step:08d}"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    try:
        lerobot_version = version("lerobot")
    except PackageNotFoundError:
        from lerobot import __version__ as lerobot_version
    with tempfile.TemporaryDirectory(prefix=".saving-", dir=folder) as staging:
        generation = Path(staging) / checkpoint.name
        native = generation / "pretrained_model"
        model.save_pretrained(native)
        preprocessor.save_pretrained(native)
        postprocessor.save_pretrained(native)
        tokenizer = next(step.input_tokenizer for step in preprocessor.steps if hasattr(step, "input_tokenizer"))
        tokenizer.save_pretrained(native / "tokenizer")
        (native / "pressb_pi05.json").write_text(json.dumps({
            "format": "pressb.pi05.v1", **wire, "checkpoint_step": step,
            "lerobot_version": lerobot_version,
        }, indent=2) + "\n")
        generation.rename(checkpoint)
    for alias in ("last", "best") if best else ("last",):
        temporary = folder / f".{alias}.partial"
        temporary.symlink_to(checkpoint.name, target_is_directory=True)
        os.replace(temporary, folder / alias)
    latest = sorted(folder.glob("step_[0-9]*"))[-keep:]
    preserved = {path.resolve() for path in [*latest, folder / "best"]}
    for path in folder.glob("step_[0-9]*"):
        if path.resolve() not in preserved:
            shutil.rmtree(path)
    return checkpoint / "pretrained_model"


def _make_data(config):
    from pressb.training.pi05_data import fit_stats, make_datasets

    seed = int(config.get("seed", 42))
    policy, checkpoint = policy_config(config.get("policy", {}))
    resolved = deepcopy(config)
    resolved["image_transforms"] = resolve_image_transforms_config(config)
    train, val, split, wire = make_datasets(
        config["dataset"], policy, seed=seed, transforms=resolved["image_transforms"],
        wire_config=config.get("pressb"),
    )
    set_features(policy, train, wire)
    return resolved, policy, checkpoint, train, val, split, wire, fit_stats(train.source)


def train(config):
    import torch
    import torch.distributed as dist
    from pressb.models.pi05 import initialize_pi05_device

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return _train(config, rank=0, world_size=1, local_device=None)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    requested_device = config.get("policy", {}).get("device") or "cuda"
    kind = str(requested_device).split(":", 1)[0]
    if kind not in ("cuda", "npu"):
        raise ValueError("Distributed π0.5 SFT requires CUDA or NPU")
    device = initialize_pi05_device(f"{kind}:{local_rank}")
    module = getattr(torch, kind)
    if not 0 <= local_rank < module.device_count():
        raise ValueError(f"LOCAL_RANK must select a visible {kind.upper()} device for π0.5 SFT")
    module.set_device(device)
    dist.init_process_group(backend="hccl" if kind == "npu" else "nccl", timeout=timedelta(minutes=30))
    try:
        return _train(config, rank=rank, world_size=world_size, local_device=device)
    finally:
        dist.destroy_process_group()


def _per_device_batch_size(config):
    batch_size = int(config.get("per_device_batch_size", config.get("batch_size", 4)))
    if "batch_size" in config and int(config["batch_size"]) != batch_size:
        raise ValueError("batch_size and per_device_batch_size must agree")
    return batch_size


def _accumulation_steps(config):
    value = config.get("gradient_accumulation_steps", 1)
    if type(value) is not int or value < 1:
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    return value


def _training_batches(dataset, loader, sampler):
    """Cycle epochs without discarding a partially accumulated optimizer step."""
    epoch = 0
    while True:
        dataset.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        yielded = False
        for raw in loader:
            yielded = True
            yield raw, epoch
        if not yielded:
            raise ValueError("Training loader has no batches; reduce batch size or disable drop_last")
        epoch += 1


def _optimizer_step(model, optimizer, raw_batches, preprocessor, cameras, device, *,
                    bf16, parameters, max_grad_norm, world_size):
    """One update, with sample-weighted accumulation and one DDP gradient sync.

    A final short data batch can cross an epoch boundary. Weight its mean loss
    by its actual sample count rather than treating it as a full microbatch.
    Every rank has matching sizes under DistributedSampler, including its tail.
    """
    import torch
    import torch.distributed as dist

    counts = [int(raw["action"].shape[0]) for raw in raw_batches]
    if not counts or min(counts) < 1:
        raise ValueError("Optimizer updates require nonempty microbatches")
    samples = sum(counts)
    optimizer.zero_grad(set_to_none=True)
    average_loss = None
    for index, (raw, count) in enumerate(zip(raw_batches, counts, strict=True)):
        sync = model.no_sync() if world_size > 1 and index < len(raw_batches) - 1 else nullcontext()
        with sync:
            batch = _batch(raw, cameras, preprocessor)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                loss, _ = model(batch)
            global_loss = loss.detach().float().clone()
            if world_size > 1:
                # Coordinate nonfinite-loss detection before any gradient
                # collective so all ranks fail on the same microbatch.
                dist.all_reduce(global_loss)
                global_loss.div_(world_size)
            if not torch.isfinite(global_loss):
                raise FloatingPointError("Nonfinite π0.5 training loss")
            weight = count / samples
            contribution = global_loss * weight
            average_loss = contribution if average_loss is None else average_loss + contribution
            (loss * weight).backward()
    # Keep the native CUDA/CPU fast path; unsupported NPU foreach norm kernels
    # are avoided explicitly instead of silently copying gradients to CPU.
    grad = torch.nn.utils.clip_grad_norm_(
        parameters, max_grad_norm, error_if_nonfinite=True,
        foreach=False if device.type == "npu" else None,
    )
    optimizer.step()
    return average_loss, grad, samples


def _barrier(device):
    import torch.distributed as dist

    # PyTorch's device_ids argument is an NCCL-specific barrier hint. HCCL
    # already uses the rank's explicitly selected NPU.
    dist.barrier(**({"device_ids": [device.index]} if device.type == "cuda" else {}))


def _train(config, *, rank, world_size, local_device):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler, Subset
    from pressb.models.pi05 import IO_FILENAME, create_pi05_policy, load_pi05_policy

    distributed = world_size > 1
    primary = rank == 0
    steps = int(config.get("steps", 30000))
    batch_size = _per_device_batch_size(config)
    accumulation_steps = _accumulation_steps(config)
    drop_last = config.get("drop_last", False)
    if type(drop_last) is not bool:
        raise ValueError("drop_last must be boolean")
    workers = int(config.get("num_workers", 4))
    eval_every = int(config.get("eval_every", 1000))
    save_every = int(config.get("save_every", eval_every))
    log_every = int(config.get("log_every", 50))
    keep_checkpoints = int(config.get("keep_checkpoints", 2))
    validation_samples = int(config.get("validation_samples", 256))
    if min(steps, batch_size, eval_every, save_every, log_every, keep_checkpoints) < 1 or min(workers, validation_samples) < 0:
        raise ValueError("Invalid π0.5 training or evaluation intervals")
    output = Path(config["output_dir"]).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    effective_config = config
    if distributed:
        effective_config = {**config, "policy": {**config.get("policy", {}), "device": str(local_device)}}
    resolved, policy, source_checkpoint, train_data, val_data, split, wire, stats = _make_data(effective_config)
    batch_geometry = dict(world_size=world_size, per_device_batch_size=batch_size,
                          gradient_accumulation_steps=accumulation_steps,
                          global_batch_size=batch_size * world_size * accumulation_steps)
    resolved.update(batch_geometry)
    tokenizer_path = config.get("tokenizer_path")
    if tokenizer_path is not None:
        tokenizer_path = Path(tokenizer_path).expanduser().resolve(strict=True)
    elif source_checkpoint is not None and (source_checkpoint / "tokenizer").is_dir():
        tokenizer_path = source_checkpoint / "tokenizer"
    device = torch.device(policy.device)
    if device.type in ("cuda", "npu"):
        module = getattr(torch, device.type)
        device = torch.device(device.type, device.index if device.index is not None else module.current_device())
        module.set_device(device)
        policy.device = str(device)
    bf16 = policy.dtype == "bfloat16"
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    preprocessor, postprocessor = make_processors(policy, stats, tokenizer_path)
    if source_checkpoint is None:
        model = create_pi05_policy(policy).to(device)
    else:
        model = load_pi05_policy(source_checkpoint, policy, device=device,
                                 pretrained_base=not (source_checkpoint / IO_FILENAME).is_file())
    training_model = model
    if distributed:
        training_model = DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index,
            find_unused_parameters=True, broadcast_buffers=False, gradient_as_bucket_view=True,
        )
        # Construct identical initial models, then sample independent flow noise on each rank.
        torch.manual_seed(seed + rank)
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_cfg = policy.get_optimizer_preset()
    optimizer = optimizer_cfg.build(parameters)
    scheduler = policy.get_scheduler_preset().build(optimizer, steps)
    cameras = list(wire["camera_keys"].values())
    loader_kwargs = dict(batch_size=batch_size, num_workers=workers, pin_memory=device.type == "cuda")
    if workers:
        loader_kwargs.update(persistent_workers=True, multiprocessing_context="spawn")
    sampler = (DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
               if distributed else None)
    train_loader = DataLoader(train_data, sampler=sampler, shuffle=sampler is None,
                              generator=torch.Generator().manual_seed(seed + rank), drop_last=drop_last,
                              **loader_kwargs)
    if primary:
        if 0 < validation_samples < len(val_data):
            val_data = Subset(val_data, torch.linspace(0, len(val_data) - 1, validation_samples).long().tolist())
        val_loader = DataLoader(val_data, **loader_kwargs)
        output.mkdir(parents=True, exist_ok=False)
        (output / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n")
        (output / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    if distributed:
        _barrier(device)
    started = time.monotonic()
    step, best_loss = 0, float("inf")
    batches = _training_batches(train_data, train_loader, sampler)
    log_context = (output / "metrics.jsonl").open("w", buffering=1) if primary else nullcontext()
    with log_context as log:
        while step < steps:
            training_model.train()
            group = [next(batches) for _ in range(accumulation_steps)]
            epoch = group[-1][1]
            global_loss, grad, local_samples = _optimizer_step(
                training_model, optimizer, [raw for raw, _ in group], preprocessor, cameras, device,
                bf16=bf16, parameters=parameters, max_grad_norm=optimizer_cfg.grad_clip_norm or float("inf"),
                world_size=world_size,
            )
            del group
            scheduler.step()
            step += 1
            do_eval = step % eval_every == 0 or step == steps
            do_save = step % save_every == 0 or step == steps
            if primary:
                val_loss, improved = None, False
                if do_eval:
                    val_loss = validate(model, val_loader, preprocessor, cameras, device, seed + 10000, bf16=bf16)
                    improved = val_loss < best_loss
                    best_loss = min(val_loss, best_loss)
                if step == 1 or step % log_every == 0 or do_eval:
                    record = dict(step=step, epoch=epoch, loss=float(global_loss), validation_loss=val_loss,
                                  grad_norm=float(grad), lr=scheduler.get_last_lr()[0], **batch_geometry,
                                  global_batch_samples=local_samples * world_size,
                                  elapsed_s=time.monotonic() - started)
                    line = json.dumps(record, allow_nan=False)
                    print(line, flush=True)
                    log.write(line + "\n")
                if improved or do_save:
                    _save_checkpoint(output, step, model, preprocessor, postprocessor, wire,
                                     best=improved, keep=keep_checkpoints)
            if distributed and (do_eval or do_save):
                _barrier(device)
    if primary:
        return dict(step=step, output_dir=str(output), best_validation_loss=best_loss, **batch_geometry,
                    checkpoint=str(output / "checkpoints/last/pretrained_model"))


def main(argv=None):
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true", help="Check split, data and normalization without model weights")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if args.prepare_only:
        _, policy, _, train_data, val_data, split, _, _ = _make_data(config)
        for data in (train_data, val_data):
            if data[0]["action"].shape[0] != policy.chunk_size:
                raise ValueError("LeRobot action chunk differs from π0.5 policy horizon")
        result = dict(train_frames=len(train_data), validation_frames=len(val_data), split=split,
                      chunk_size=policy.chunk_size)
    else:
        result = train(config)
    if result is not None:
        print(json.dumps(result, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
