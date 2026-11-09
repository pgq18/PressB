"""Train the bundled visual Diffusion Policy on an episode-split LeRobot dataset."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from pressb.model_sources.diffusion_policy.policy import VisualDiffusionPolicy
from pressb.training.diffusion_policy_data import LeRobotData, DiffusionDataset
from pressb.training.image_transforms import resolve_image_transforms_config


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, device="cpu", *, use_ema=True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "pressb.diffusion_policy.v1":
        raise ValueError("Not a PressB visual Diffusion Policy checkpoint")
    model = VisualDiffusionPolicy(checkpoint["model_config"], checkpoint["normalization"], device=device)
    model.load_state_dict(checkpoint["ema"] if use_ema else checkpoint["model"], strict=True)
    return model, checkpoint


def _device_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def validate(model, loader, device, seed=10000):
    model.eval()
    losses, counts = [], []
    # Validation images are unaugmented. Fixed latent noise and timesteps make
    # checkpoint selection comparable rather than adding Monte Carlo drift.
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch in loader:
        batch = _device_batch(batch, device)
        b = len(batch["state"])
        t = torch.randint(model.diffusion.denoising_steps, (b,), device=device, generator=generator)
        noise = torch.randn(batch["actions"].shape, device=device, generator=generator)
        loss = model.loss(batch["images"], batch["state"], batch["task"], batch["actions"], timesteps=t, noise=noise)
        losses.append(float(loss) * b)
        counts.append(b)
    return sum(losses) / sum(counts)


def train(config):
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get("cpu_threads", 8)))
    device = torch.device(config.get("device", "cuda:0"))
    precision = config.get("precision", "float32")
    if precision not in ("float32", "bfloat16"):
        raise ValueError("precision must be float32 or bfloat16")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    config = deepcopy(config)
    config["image_transforms"] = resolve_image_transforms_config(config)
    output = Path(config["output_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and not config.get("resume"):
        raise FileExistsError("output_dir already contains a checkpoint; use explicit resume or a new output_dir")
    data = LeRobotData(config["dataset"])
    data.prepare_images(config.get("image_cache_dir", str(output / "rgb_cache")))
    split = data.split(seed, float(config.get("validation_fraction", .1)))
    normalization = data.normalization(split["train"])
    model_config = deepcopy(config["model"])
    model_config.update(tasks=data.task_names, cameras=data.cameras, resolution=data.resolution,
                        control_hz=data.fps)
    train_data = DiffusionDataset(data, split["train"], model_config["horizon"], seed=seed,
                                 image_transforms=config["image_transforms"], stride=config.get("sample_stride", 1))
    val_data = DiffusionDataset(data, split["val"], model_config["horizon"], split="val", seed=seed,
                               max_samples=int(config.get("validation_samples", 1024)))
    workers = int(config.get("num_workers", 4))
    batch_size = int(config.get("batch_size", 128))
    loader_options = dict(num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            generator=torch.Generator().manual_seed(seed ^ 0x5DEECE66D), **loader_options)
    model = VisualDiffusionPolicy(model_config, normalization, device=device)
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-4)),
                                  weight_decay=float(config.get("weight_decay", 1e-6)))
    steps = int(config.get("max_steps", 30000))
    warmup = int(config.get("warmup_steps", 500))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:
        min((s + 1) / max(warmup, 1), 1.) * .5 * (1 + math.cos(math.pi * min(max(s - warmup, 0) / max(steps - warmup, 1), 1))))
    effective_training = {key: value for key, value in config.items()
                          if key not in {"resume", "output_dir", "image_cache_dir", "device", "dataset", "num_workers", "cpu_threads"}}
    preprocessing = {key: value for key, value in config["dataset"].items() if key != "root"}
    contract = dict(model_config=model_config, normalization=normalization, dataset_sha256=data.sha256,
                    split=split, image_transforms=config["image_transforms"], seed=seed,
                    image_transform_order=list(config["image_transforms"]["tfs"]),
                    preprocessing=preprocessing, effective_training=effective_training)
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    step, epoch, cursor, best = 0, 0, 0, float("inf")
    if config.get("resume"):
        previous = torch.load(config["resume"], map_location="cpu", weights_only=True)
        if previous.get("training_fingerprint") != fingerprint:
            raise ValueError("Resume dataset/model/normalization/split/augmentation contract differs")
        model.load_state_dict(previous["model"])
        ema.load_state_dict(previous["ema"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        step, epoch, cursor, best = previous["step"], previous["epoch"], previous["cursor"], previous["best_val_loss"]
        torch.set_rng_state(previous["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(previous["cuda_rng"], device)
    # Publish provenance only after any resume contract is accepted.
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "split.json").write_text(json.dumps(dict(dataset_sha256=data.sha256, **split), indent=2) + "\n")
    started = time.monotonic()
    log = (output / "metrics.jsonl").open("a")
    last_val = None
    try:
        while step < steps:
            train_data.set_epoch(epoch)
            indices = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
            train_loader = DataLoader(train_data, batch_size=batch_size, sampler=indices[cursor:],
                generator=torch.Generator().manual_seed(seed + epoch), **loader_options)
            for batch in train_loader:
                model.train()
                batch = _device_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=precision == "bfloat16"):
                    loss = model.loss(batch["images"], batch["state"], batch["task"], batch["actions"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite diffusion SFT loss")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.)))
                optimizer.step()
                scheduler.step()
                step += 1
                cursor += len(batch["state"])
                decay = min(float(config.get("ema_decay", .999)), 1 - (1 + step) ** -.75)
                with torch.no_grad():
                    for target, source in zip(ema.parameters(), model.parameters()):
                        target.lerp_(source, 1 - decay)
                    for target, source in zip(ema.buffers(), model.buffers()):
                        target.copy_(source)
                do_eval = step % int(config.get("eval_every", 500)) == 0 or step == steps
                improved = False
                if do_eval:
                    last_val = validate(ema, val_loader, device)
                    improved = last_val < best
                    best = min(last_val, best)
                if step % int(config.get("log_every", 25)) == 0 or do_eval or step == 1:
                    record = dict(step=step, epoch=epoch, loss=float(loss), validation_loss=last_val,
                                  best_validation_loss=best if np.isfinite(best) else None,
                                  gradient_norm=float(gradient_norm), lr=scheduler.get_last_lr()[0],
                                  elapsed_s=time.monotonic() - started)
                    line = json.dumps(record)
                    print(line, flush=True)
                    log.write(line + "\n")
                    log.flush()
                if do_eval or step % int(config.get("save_every", 2000)) == 0:
                    payload = dict(format="pressb.diffusion_policy.v1", model_config=model_config,
                                   normalization=normalization, model=model.state_dict(), ema=ema.state_dict(),
                                   optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                   step=step, epoch=epoch, cursor=cursor, best_val_loss=best,
                                   torch_rng=torch.get_rng_state(), cuda_rng=(torch.cuda.get_rng_state(device) if device.type == "cuda" else None),
                                   training_fingerprint=fingerprint, training_contract=contract, config=config)
                    save_checkpoint(output / "last.pt", payload)
                    if improved:
                        save_checkpoint(output / "best.pt", payload)
                    if step % int(config.get("milestone_every", 100000)) == 0:
                        save_checkpoint(output / f"step_{step:08d}.pt", payload)
                if step >= steps:
                    break
            epoch += 1
            cursor = 0
    finally:
        log.close()
    return dict(output_dir=str(output), step=step, best_validation_loss=best)


def main(argv=None):
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if args.prepare_only:
        data = LeRobotData(config["dataset"])
        data.prepare_images(config["image_cache_dir"])
        print(json.dumps(dict(dataset_sha256=data.sha256, frames=len(data.states), tasks=data.task_names)))
        return 0
    print(json.dumps(train(config)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
