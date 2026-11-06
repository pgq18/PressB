"""Independent joint7 training for a paired-layout, three-instruction dataset.

Run with the same --config / --resume / --probe-steps CLI as train_table_buttons.
The original single-task trainer and its nine runs remain unchanged. Shared
initialization, parameter configuration, weighted masking and checkpoint helpers
are imported, while this entrypoint owns the multi-task data and validation loop.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from starVLA.training import train_piper as shared
from starVLA.training.train_table_buttons import (
    SCHEMA, GRIPPER_SCORE_WEIGHT, setup_distributed, load_joint7_pretrained,
    joint_metric_sums, summarize_joint_metrics, reconcile_best_checkpoint,
    save_checkpoint,
)

COLORS = ("red", "yellow", "blue")


def validate_anchor_balance(dataset, anchors):
    expected = {color: 50 for color in COLORS}
    actual = {color: 0 for color in COLORS}
    for episode, frame in anchors:
        actual[dataset._episodes[episode]["target_color"]] += 1
    if len(anchors) != 150 or len(set(anchors)) != 150 or actual != expected:
        raise ValueError("Expected 150 distinct validation anchors, 10 episodes x 5 per color")


def summarize_multitask_metrics(totals):
    """Report every task and select using equal weight for the three colors."""
    totals = np.asarray(totals, dtype=np.float64)
    if totals.shape != (3, 12) or not np.isfinite(totals).all() or np.any(totals[:, 0] <= 0):
        raise ValueError("Expected finite, nonempty validation totals for all three colors")
    def result(row):
        metrics = summarize_joint_metrics(row)
        metrics.update(flow_loss=float(row[8] / max(row[9], 1)),
                       wm_loss_weighted_0_1=float(row[10] / max(row[11], 1)))
        return metrics
    per_color = {color: result(totals[i]) for i, color in enumerate(COLORS)}
    overall = result(totals.sum(axis=0))
    overall["selection_score"] = float(np.mean([value["selection_score"] for value in per_color.values()]))
    overall["selection_aggregation"] = "equal-weight mean of three per-color first-action scores"
    return overall, per_color


@torch.no_grad()
def evaluate(model, dataset, anchors, rank, world, device, seed, step, output):
    model.eval()
    totals = np.zeros((3, 12), dtype=np.float64)
    started = time.monotonic()
    if rank == 0:
        print(json.dumps(dict(event="validation_started", step=step, anchors=len(anchors))), flush=True)
    for ordinal in range(rank, len(anchors), world):
        episode, frame = anchors[ordinal]
        example = dataset.get_frame(episode, frame)
        color = COLORS.index(example["target_color"])
        # Identical held-out layouts use identical diffusion noise for each language.
        example_seed = seed + example["original_source_episode_id"] * 100003 + frame
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(example_seed)
            torch.cuda.manual_seed(example_seed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = model.predict_action([example["image"]], [example["lang"]],
                    state=np.stack([example["state"]]))["normalized_actions"][0]
                losses = model([example])
        values = joint_metric_sums(predicted, example["action"], example["state"], example["action_is_pad"])
        future = (~example["video_is_pad"].reshape(-1, model.vj_encoder.config.tubelet_size).any(-1))[1:].sum()
        totals[color, :8] += values
        totals[color, 8:] += [float(losses["action_loss"]) * values[1], values[1],
                              float(losses["wm_loss"]) * future, future]
    if not np.isfinite(totals).all():
        raise RuntimeError("Nonfinite multi-task validation values")
    reduced = torch.as_tensor(totals, device=device)
    dist.all_reduce(reduced)
    overall, per_color = summarize_multitask_metrics(reduced.cpu().numpy())
    if overall["anchors"] != len(anchors) or any(m["anchors"] != 50 for m in per_color.values()):
        raise RuntimeError("Validation skipped or repeated per-color anchors")
    result = dict(step=step, evaluation_seed=seed, seconds=time.monotonic() - started,
                  overall=overall, per_color=per_color, group=dataset.group,
                  selection="mean over red/yellow/blue of first_joint_mae_deg + 22.5 * first_gripper_mae_cm",
                  interpretation="offline held-out-layout action prediction; not simulator task success rate")
    if rank == 0:
        shared.write_json(output / f"validation_{step:06d}.json", result)
        print(json.dumps(dict(event="validation", step=step, **overall, per_color=per_color)), flush=True)
    return result


def build_contract(cfg, normalization):
    contract, _ = shared.training_contract(cfg)
    contract.update(schema=SCHEMA, multitask=True, normalization=normalization,
                    target_colors=list(COLORS), validation_selection_weight=GRIPPER_SCORE_WEIGHT,
                    validation_selection_aggregation="equal weight per color",
                    validation_anchors_per_color=50,
                    split_rule="shared original episode permutation seed42: first10 layouts validation across all colors, remaining90 train",
                    split_counts={"train": 270, "val": 30},
                    language="per-episode Press the <color> button., identical across distance profiles",
                    initialization="official pretrained, four joint7 I/O tensors reset")
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return contract, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", type=Path, help="Resume this same joint7 run's checkpoints/last")
    parser.add_argument("--probe-steps", type=int, default=0, help="Resource probe: run N optimizer updates, no checkpoints or validation")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    if args.probe_steps < 0 or (args.probe_steps and args.resume):
        raise ValueError("Probe requires positive steps and fresh initialization")
    from omegaconf import OmegaConf
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.set))
    if args.run_id:
        cfg.run_id = args.run_id
    if cfg.framework.action_model.action_dim != 7 or cfg.framework.action_model.state_dim != 7:
        raise ValueError("This trainer requires joint7 state/action interfaces")
    if int(cfg.seed) != 42:
        raise ValueError("These three multi-task runs use the shared layout split seed42")
    rank, world, local_rank, device = setup_distributed()
    cfg.framework.qwenvl.device_map = f"cuda:{local_rank}"
    memory_limit = float(cfg.trainer.memory_limit_gib) * 1024**3
    torch.cuda.set_per_process_memory_fraction(min(1., memory_limit / torch.cuda.get_device_properties(device).total_memory), device)
    shared.seed_all(int(cfg.seed))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output = Path(cfg.run_root_dir) / cfg.run_id
    if rank == 0:
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise FileExistsError(f"Run exists: {output}; use a fresh run-id or --resume")
        output.mkdir(parents=True, exist_ok=True)
        shared.write_json(output / "status.json", dict(status="initializing", probe=bool(args.probe_steps)))
    dist.barrier()
    from starVLA.dataloader.table_buttons_multitask_lerobot import TableButtonsMultitaskLeRobotDataset, collate_fn
    common = dict(root=cfg.datasets.vla_data.data_root_dir, split_manifest=output / "split.json", action_horizon=7,
                  video_horizon=8, resolution=int(cfg.datasets.vla_data.resolution_size), seed=int(cfg.seed),
                  frame_cache_size=int(cfg.datasets.vla_data.frame_cache_size), decoder_cache_size=int(cfg.datasets.vla_data.decoder_cache_size))
    train_data = TableButtonsMultitaskLeRobotDataset(split="train", sample_stride=int(cfg.datasets.vla_data.sample_stride), **common)
    val_data = TableButtonsMultitaskLeRobotDataset(split="val", **common)
    anchors = shared.validation_anchors(val_data, int(cfg.trainer.val_anchors_per_episode), int(cfg.trainer.val_max_samples))
    validate_anchor_balance(val_data, anchors)
    batch_size, accumulation = int(cfg.datasets.vla_data.per_device_batch_size), int(cfg.trainer.gradient_accumulation_steps)
    geometry = shared.epoch_geometry(len(train_data), world, batch_size, accumulation, int(cfg.trainer.target_epochs))
    cfg.trainer.max_train_steps = geometry["target_steps"]
    cfg.trainer.num_warmup_steps = max(1, round(.05 * geometry["target_steps"]))
    maximum = args.probe_steps or geometry["target_steps"]
    contract, fingerprint = build_contract(OmegaConf.to_container(cfg, resolve=True), train_data.normalization)
    provenance = dict(dataset_sha256=train_data.export_manifest_sha256, split_sha256=train_data.split_manifest_sha256,
                      dataset_fingerprint_sha256=train_data.fingerprint_sha256, training_fingerprint=fingerprint,
                      dataset_group=train_data.group, multitask=True, target_colors=list(COLORS))
    progress = dict(step=0, epoch=0, batch_in_epoch=0, best_score=None)
    training, checkpoint_dir = None, None
    if args.resume:
        checkpoint_dir = args.resume.resolve(strict=True)
        if checkpoint_dir.parent != (output / "checkpoints").resolve():
            raise ValueError("Resume is restricted to this run's own checkpoint directory")
        manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
        if manifest.get("schema") != SCHEMA or manifest.get("kind") != "resumable_last" or not manifest.get("complete"):
            raise ValueError("Resume requires a complete joint7 last checkpoint")
        for key, value in provenance.items():
            if manifest.get(key) != value:
                raise ValueError(f"Resume contract changed: {key}")
        for name in ("model", "training"):
            if (checkpoint_dir / f"{name}.pt").stat().st_size != manifest[f"{name}_bytes"]:
                raise ValueError(f"Incomplete {name} checkpoint")
        training = torch.load(checkpoint_dir / "training.pt", map_location="cpu", weights_only=False, mmap=True)
        if training.get("schema") != SCHEMA or training["world_size"] != world:
            raise ValueError("Resume world size/schema changed")
        for key in (*provenance, *progress):
            if training.get(key) != manifest.get(key):
                raise ValueError(f"Training/manifest mismatch: {key}")
        if rank == 0:
            reconcile_best_checkpoint(output / "checkpoints", checkpoint_dir, manifest)
        dist.barrier()
        progress = {key: training[key] for key in progress}
        shared.normalize_epoch_cursor(progress, geometry["batches_per_epoch"])
        expected_step = progress["epoch"] * geometry["steps_per_epoch"] + math.ceil(progress["batch_in_epoch"] / accumulation)
        if progress["step"] != expected_step or progress["batch_in_epoch"] % accumulation:
            raise ValueError("Resume cursor does not match epoch geometry")
        if maximum <= progress["step"]:
            raise ValueError("Checkpoint already meets this run's training target")
    if rank == 0:
        shared.write_json(output / "epoch_geometry.json", geometry)
        shared.write_json(output / "normalization.json", train_data.normalization)
        shared.write_json(output / "training_contract.json", dict(contract=contract, fingerprint=fingerprint))
        shared.write_json(output / "validation_anchors.json", dict(anchors=anchors, **provenance))
        shared.write_json(output / "dataset_fingerprint.json", train_data.fingerprint)
        print(json.dumps(dict(event="dataset_ready", group=train_data.group, train_episodes=len(train_data.source_episode_ids),
                              validation_episodes=len(val_data.source_episode_ids), **geometry)), flush=True)
    from starVLA.model.framework.VLA_JEPA import VLA_JEPA
    model = VLA_JEPA(config=cfg)
    if checkpoint_dir:
        from transformers import AutoProcessor
        from starVLA.inference.piper_policy import install_saved_processor
        install_saved_processor(model, AutoProcessor.from_pretrained(output / "processor", local_files_only=True))
        loading = dict(initialization="resume_this_joint7_run", checkpoint=str(checkpoint_dir), source_step=progress["step"])
    else:
        loading = load_joint7_pretrained(model, cfg.trainer.pretrained_checkpoint)
        loading["checkpoint_path"] = str(Path(cfg.trainer.pretrained_checkpoint).resolve())
        loading["checkpoint_sha256"] = cfg.trainer.get("pretrained_sha256") or shared.sha256(cfg.trainer.pretrained_checkpoint)
    model.to(device)
    frozen, parameter_report = shared.configure_parameters(model, cfg.trainer.enable_gradient_checkpointing)
    if checkpoint_dir:
        weights = torch.load(checkpoint_dir / "model.pt", map_location="cpu", weights_only=True, mmap=True)
        shared.validate_tied_source(model, weights)
        for name, tensor in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False)):
            if name in weights and tensor.dtype != weights[name].dtype:
                tensor.data = tensor.data.to(weights[name].dtype)
        model.load_state_dict(weights, strict=True)
        del weights
    groups = [dict(params=[p for name, p in model.named_parameters() if name.startswith(prefix) and p.requires_grad],
                   lr=float(cfg.trainer.learning_rate[prefix.rstrip(".")]), name=prefix.rstrip("."))
              for prefix in ("qwen_vl_interface.", "vj_predictor.", "action_model.")]
    if sum(p.numel() for group in groups for p in group["params"]) != parameter_report["trainable_numel"]:
        raise RuntimeError("Optimizer groups omit/duplicate trainable parameters")
    optimizer = torch.optim.AdamW(groups, betas=tuple(cfg.trainer.optimizer.betas), eps=float(cfg.trainer.optimizer.eps),
                                 weight_decay=float(cfg.trainer.weight_decay), foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, shared.lr_functions(geometry["target_steps"], int(cfg.trainer.num_warmup_steps)))
    resume_rng = None
    if training:
        optimizer.load_state_dict(training["optimizer"])
        scheduler.load_state_dict(training["scheduler"])
        if scheduler.last_epoch != progress["step"]:
            raise ValueError("Restored scheduler step mismatch")
        steps = [int(state["step"]) for state in optimizer.state.values() if "step" in state]
        if not steps or any(step != progress["step"] for step in steps):
            raise ValueError("Restored optimizer step mismatch")
        for group, rate in zip(optimizer.param_groups, scheduler.get_last_lr()):
            group["lr"] = rate
        resume_rng = training["rng_by_rank"][rank]
        del training
    probes = shared.parameter_probes(model)
    if rank == 0:
        shared.write_json(output / ("resume_loading.json" if args.resume else "pretrained_loading.json"), loading)
        shared.write_json(output / "parameters.json", parameter_report)
        shared.write_json(output / "action_schema.json", dict(schema=SCHEMA, state_dim=7, action_dim=7,
            action_horizon=7, video_horizon=8, fps=30, representation="absolute joint targets; joint1..6 + full_gripper_width",
            dataset_units=["deg"] * 6 + ["cm"], normalization=train_data.normalization, cameras=["up", "wrist"], **provenance))
        if not args.resume:
            saved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            saved_cfg.framework.qwenvl.device_map = "cuda"
            OmegaConf.save(saved_cfg, output / "config.yaml")
            model.qwen_vl_interface.processor.save_pretrained(output / "processor")
        print(json.dumps(dict(event="initialized", pretrained=loading, **parameter_report)), flush=True)
    wrapped = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False,
                  broadcast_buffers=False, gradient_as_bucket_view=True) if world > 1 else model
    sampler = shared.OffsetDistributedSampler(train_data, num_replicas=world, rank=rank, shuffle=True, seed=int(cfg.seed), drop_last=False)
    workers = int(cfg.datasets.vla_data.num_workers)
    loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers=workers, collate_fn=collate_fn,
                        drop_last=False, persistent_workers=workers > 0, worker_init_fn=shared.worker_seed,
                        generator=torch.Generator().manual_seed(int(cfg.seed) + rank * 1009),
                        **({"multiprocessing_context": "spawn", "prefetch_factor": 2} if workers else {}))
    shared.seed_all(int(cfg.seed) + rank * 1009)
    if resume_rng:
        shared.restore_rng(resume_rng)
    elif not args.probe_steps:
        evaluate(model, val_data, anchors, rank, world, device, int(cfg.seed) + 500000, 0, output)
    if rank == 0:
        shared.write_json(output / "initialization.json", dict(step=progress["step"], **geometry,
            global_batch_size=world * batch_size * accumulation, probe=bool(args.probe_steps), **provenance))
    optimizer.zero_grad(set_to_none=True)
    started, checked_gradients = time.monotonic(), False
    while progress["step"] < maximum:
        sampler.set_epoch(progress["epoch"])
        sampler.set_start_index(progress["batch_in_epoch"] * batch_size)
        model.train()
        for module in frozen:
            module.eval()
        iterator = iter(loader)
        while progress["batch_in_epoch"] < geometry["batches_per_epoch"] and progress["step"] < maximum:
            step_start = time.monotonic()
            microbatches = min(accumulation, geometry["batches_per_epoch"] - progress["batch_in_epoch"])
            batches = [next(iterator) for _ in range(microbatches)]
            weights = shared.loss_weights(batches, model.vj_encoder.config.tubelet_size, device, world)
            logged = torch.zeros(2, device=device)
            for micro, batch in enumerate(batches):
                context = wrapped.no_sync() if world > 1 and micro + 1 < microbatches else nullcontext()
                with context:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        losses = wrapped(batch)
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
                    raise RuntimeError(f"Trainable parameters missing from loss graph: {missing}")
                checked_gradients = True
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                  float(cfg.trainer.max_grad_norm), error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            progress["step"] += 1
            epoch_finished = shared.normalize_epoch_cursor(progress, geometry["batches_per_epoch"])
            dist.all_reduce(logged)
            torch.cuda.synchronize(device)
            record = dict(event="train", **progress, action_loss=float(logged[0] / world),
                          wm_loss_weighted_0_1=float(logged[1] / world), grad_norm=float(norm),
                          learning_rates=scheduler.get_last_lr(), seconds=time.monotonic() - started,
                          step_seconds=time.monotonic() - step_start, peak_memory_gib=torch.cuda.max_memory_allocated(device) / 1024**3)
            if rank == 0:
                with (output / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                shared.write_json(output / "status.json", dict(status="training", target_steps=maximum,
                                  group=train_data.group, probe=bool(args.probe_steps), **record))
                if progress["step"] == 1 or progress["step"] % int(cfg.trainer.logging_frequency) == 0 or args.probe_steps:
                    print(json.dumps(record), flush=True)
            if not args.probe_steps and (epoch_finished or progress["step"] % int(cfg.trainer.eval_interval) == 0 or progress["step"] == maximum):
                metrics = evaluate(model, val_data, anchors, rank, world, device, int(cfg.seed) + 500000, progress["step"], output)
                score = metrics["overall"]["selection_score"]
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
        parameters, changes = dict(model.named_parameters()), {}
        for name, before in probes.items():
            after = parameters[name].detach().flatten()[:len(before)].float().cpu()
            diff = (after - before).abs()
            changes[name] = dict(elements=len(before), changed_elements=int((diff != 0).sum()),
                                 max_abs_change=float(diff.max()), mean_abs_change=float(diff.mean()))
        shared.write_json(output / "parameter_update_evidence.json", dict(final_step=progress["step"], probes=changes))
        completed = dict(success=True, **progress, **provenance, seconds=time.monotonic() - started,
                         peak_memory_gib=torch.cuda.max_memory_allocated(device) / 1024**3, world_size=world)
        if args.probe_steps:
            completed.update(probe=True, batch_size=batch_size, accumulation=accumulation)
            shared.write_json(output / "probe.json", completed)
        else:
            completed.update(best_checkpoint=str((output / "checkpoints/best").resolve()),
                             last_checkpoint=str((output / "checkpoints/last").resolve()))
            shared.write_json(output / "completed.json", completed)
        shared.write_json(output / "status.json", dict(status="probe_complete" if args.probe_steps else "completed", **completed))
        print(json.dumps(dict(event="probe_complete" if args.probe_steps else "completed", **completed)), flush=True)
    train_data.close()
    val_data.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
