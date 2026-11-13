"""Independent joint-space VLA-JEPA fine-tuning for one table-buttons dataset.

The elevator trainer remains unchanged. Shared DDP, masking, scheduling and
parameter-freezing helpers come from train_piper, while all data, metrics and
checkpoint contracts describe either six arm actions or six angles plus gripper.
Both modes retain seven observed state coordinates; joint7 remains the default.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from starVLA.training import train_piper as shared

INTERFACE_KEYS = shared.ADAPTED_KEYS
SCALE = np.array([180.] * 6 + [8.], dtype=np.float64)
GRIPPER_SCORE_WEIGHT = 180. / 8.
SCHEMA = "vlajepa.table_buttons.joint7.v1"
JOINT6_SCHEMA = "vlajepa.table_buttons.joint6.v1"
RESUMABLE_CHECKPOINT_POLICY = "resumable_best_and_last"
MODEL_ONLY_CHECKPOINT_POLICY = "best_and_final_model_only"


def configured_checkpoint_policy(cfg, resume=None):
    policy = cfg["trainer"].get("checkpoint_policy", RESUMABLE_CHECKPOINT_POLICY)
    if policy not in (RESUMABLE_CHECKPOINT_POLICY, MODEL_ONLY_CHECKPOINT_POLICY):
        raise ValueError(f"Unknown checkpoint policy: {policy}")
    if resume is not None and policy == MODEL_ONLY_CHECKPOINT_POLICY:
        raise ValueError("best_and_final_model_only does not save optimizer/RNG state and cannot resume")
    return policy


def warm_start_configuration(cfg, resume=None):
    """Explicit model-only continuation; optimizer/schedule restart in a new run."""
    trainer = cfg["trainer"]
    keys = ("warm_start_checkpoint", "warm_start_sha256", "warm_start_step")
    present = [key for key in keys if key in trainer]
    if not present:
        return None
    if len(present) != len(keys):
        raise ValueError("Warm start requires checkpoint, SHA256 and source step together")
    if resume is not None:
        raise ValueError("Model-only warm start and exact --resume are mutually exclusive")
    checkpoint, sha, step = (trainer[key] for key in keys)
    if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute() or Path(checkpoint).name != "model.pt":
        raise ValueError("warm_start_checkpoint must be an absolute model.pt path")
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("warm_start_sha256 must be a lowercase SHA256 digest")
    if type(step) is not int or step < 1:
        raise ValueError("warm_start_step must be a positive completed optimizer step")
    return dict(zip(keys, (checkpoint, sha, step)))


def inspect_warm_start(cfg, *, schema="vlajepa.joint7.v1", provenance=None, normalization=None, output=None):
    """CPU-only validation shared by launcher preflight and the actual trainer."""
    settings = warm_start_configuration(cfg)
    if settings is None:
        return None
    if schema != "vlajepa.joint7.v1":
        raise ValueError("Model-only warm start is supported only by generic joint7 SFT")
    path = Path(settings["warm_start_checkpoint"]).resolve(strict=True)
    checkpoint = path.parent
    if path.name != "model.pt" or checkpoint.parent.name != "checkpoints":
        raise ValueError("Warm start requires run/checkpoints/<generation>/model.pt")
    source_run = path.parents[2]
    if output is not None and Path(output).resolve() == source_run:
        raise ValueError("Warm start must write to a new run; the source run is immutable")
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    step = settings["warm_start_step"]
    if (manifest.get("complete") is not True or manifest.get("schema") != schema
            or type(manifest.get("step")) is not int or manifest["step"] != step):
        raise ValueError("Warm-start manifest schema, completion or source step differs")
    if path.stat().st_size != manifest.get("model_bytes"):
        raise ValueError("Warm-start model size differs from complete manifest")
    sha = shared.sha256(path)
    if sha != settings["warm_start_sha256"]:
        raise ValueError("Warm-start model SHA256 differs from the configured source")
    source_schema = json.loads((source_run / "action_schema.json").read_text())
    if (source_schema.get("schema") != schema or source_schema.get("state_dim") != 7
            or source_schema.get("action_dim") != 7):
        raise ValueError("Warm-start source action/state schema differs from generic joint7")
    for key in ("dataset_sha256", "split_sha256", "dataset_fingerprint_sha256", "task_id"):
        if provenance is not None and key in provenance:
            if manifest.get(key) != provenance[key] or source_schema.get(key) != provenance[key]:
                raise ValueError(f"Warm-start source data contract differs: {key}")
    split = json.loads((source_run / "split.json").read_text())
    split_fingerprint = hashlib.sha256(json.dumps(split, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (split_fingerprint != manifest.get("split_sha256") or source_schema.get("split_sha256") != split_fingerprint
            or split.get("source_export_manifest_sha256") != manifest.get("dataset_sha256")
            or split.get("task_id") != manifest.get("task_id")
            or split.get("camera_keys") != ["observation.images.up", "observation.images.wrist"]):
        raise ValueError("Warm-start source split, dataset identity or camera order differs")
    source_normalization = json.loads((source_run / "normalization.json").read_text())
    if source_schema.get("normalization") != source_normalization:
        raise ValueError("Warm-start source normalization metadata is inconsistent")
    if normalization is not None and source_normalization != normalization:
        raise ValueError("Warm-start source normalization differs")
    source_contract = json.loads((source_run / "training_contract.json").read_text())
    training_fingerprint = hashlib.sha256(json.dumps(source_contract["contract"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (source_contract.get("fingerprint") != training_fingerprint
            or manifest.get("training_fingerprint") != training_fingerprint
            or source_schema.get("training_fingerprint") != training_fingerprint):
        raise ValueError("Warm-start source training fingerprint is inconsistent")
    if source_contract["contract"].get("normalization") != source_normalization:
        raise ValueError("Warm-start source contract normalization is inconsistent")
    processor = source_run / "processor"
    if not processor.is_dir() or not any(processor.iterdir()):
        raise ValueError("Warm-start source saved processor is missing")
    return dict(checkpoint_path=str(path), checkpoint_sha256=sha, checkpoint_bytes=path.stat().st_size,
                processor_path=str(processor), source_run=str(source_run), source_step=step, source_schema=schema,
                manifest_sha256=shared.sha256(manifest_path), source_training_fingerprint=training_fingerprint,
                source_dataset_sha256=manifest.get("dataset_sha256"), source_split_sha256=manifest.get("split_sha256"),
                source_dataset_fingerprint_sha256=manifest.get("dataset_fingerprint_sha256"),
                optimizer_state_loaded=False, scheduler_state_loaded=False, interfaces_reinitialized=False)


def load_model_only_warm_start(model, source):
    """Restore every trained tensor and token mapping; never reset joint interfaces."""
    from transformers import AutoProcessor
    from starVLA.inference.piper_policy import install_saved_processor, load_trained_state
    processor = AutoProcessor.from_pretrained(source["processor_path"], local_files_only=True)
    install_saved_processor(model, processor)
    count = load_trained_state(model, source["checkpoint_path"])
    return dict(initialization="model_only_warm_start", **source, loaded_keys=count,
                step_zero_interfaces=interface_evidence(model))


def schema_for(action_dim):
    if action_dim not in (6, 7):
        raise ValueError("Only joint6 or joint7 action interfaces are supported")
    return JOINT6_SCHEMA if action_dim == 6 else SCHEMA


def selection_description(action_dim):
    return "first_joint_mae_deg" if action_dim == 6 else "first_joint_mae_deg + 22.5 * first_gripper_mae_cm"


def select_joint7_pretrained(source, target):
    """Compatibility API for the original single/multitask joint7 trainers."""
    return select_joint_pretrained(source, target, action_dim=7)


def select_joint_pretrained(source, target, *, action_dim):
    """Reinitialize all four interfaces, including equal-shaped action tensors."""
    schema_for(action_dim)
    if not isinstance(source, dict) or not source:
        raise ValueError("Expected nonempty official tensor state_dict")
    if all(name.startswith("module.") for name in source):
        source = {name[7:]: value for name, value in source.items()}
    if set(source) != set(target) or not all(isinstance(v, torch.Tensor) for v in source.values()):
        raise ValueError("Official checkpoint keys/types differ from the target architecture")
    loaded, adapted = {}, []
    for name, value in source.items():
        if name in INTERFACE_KEYS:
            dimension = 1 if name.endswith("layer1.weight") else 0
            expected = list(target[name].shape)
            target_dim = 7 if ".state_encoder." in name else action_dim
            if expected[dimension] != target_dim:
                raise ValueError(f"Unexpected target interface dimension: {name}, expected {target_dim}")
            expected[dimension] = 8 if ".state_encoder." in name else 7
            if list(value.shape) != expected:
                raise ValueError(f"Unexpected official interface shape: {name} {tuple(value.shape)}")
            adapted.append(dict(key=name, source_shape=list(value.shape), target_shape=list(target[name].shape),
                                reason="different action semantics: " + ("six arm joint targets" if action_dim == 6 else "joint angles and full gripper width")))
        else:
            if value.shape != target[name].shape:
                raise ValueError(f"Unapproved pretrained shape mismatch: {name}")
            loaded[name] = value
    if len(adapted) != 4:
        raise ValueError("Exactly four action/state interface tensors must be reinitialized")
    return loaded, dict(initialization=f"official_pretrained_with_new_joint{action_dim}_interfaces",
                        loaded_keys=len(loaded), loaded_numel=sum(v.numel() for v in loaded.values()),
                        adapted_keys=adapted, missing_keys=[], unexpected_keys=[],
                        components={prefix: dict(tensors=sum(n.startswith(prefix) for n in loaded),
                                    numel=sum(v.numel() for n, v in loaded.items() if n.startswith(prefix)))
                                    for prefix in shared.COMPONENTS})


def initialize_interfaces(model):
    """Use nn.Linear's initialization for only the four semantic interfaces."""
    for name in sorted(INTERFACE_KEYS):
        parameter = model.get_parameter(name)
        if name.endswith("weight"):
            torch.nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
        else:
            fan_in = model.get_parameter(name[:-4] + "weight").shape[1]
            torch.nn.init.uniform_(parameter, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))


def interface_evidence(model):
    result = {}
    for name in sorted(INTERFACE_KEYS):
        value = model.get_parameter(name).detach().float().cpu().numpy()
        result[name] = dict(shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                            mean=float(value.mean()), std=float(value.std()), max_abs=float(np.abs(value).max()))
    return result


def load_joint7_pretrained(model, path):
    return load_joint_pretrained(model, path, action_dim=7)


def load_joint_pretrained(model, path, *, action_dim):
    source = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if "state_dict" in source:
        source = source["state_dict"]
    selected, report = select_joint_pretrained(source, model.state_dict(), action_dim=action_dim)
    report["tied_parameter_groups_checked"] = shared.validate_tied_source(model, selected)
    initialize_interfaces(model)
    result = model.load_state_dict(selected, strict=False)
    if set(result.missing_keys) != INTERFACE_KEYS or result.unexpected_keys:
        raise RuntimeError("Loaded keys disagree with the joint initialization contract")
    report["step_zero_interfaces"] = interface_evidence(model)
    return report


def joint_metric_sums(prediction, target, state, padding, *, action_dim=7):
    """Physical-unit errors, excluding padded future actions, without clipping."""
    prediction, target, state = (np.asarray(v, dtype=np.float64) for v in (prediction, target, state))
    padding = np.asarray(padding)
    schema_for(action_dim)
    if prediction.shape != target.shape or prediction.shape[-1] != action_dim or padding.shape != target.shape[:-1]:
        raise ValueError("Joint validation shapes disagree")
    if padding.dtype != np.bool_ or padding[0] or not all(np.isfinite(v).all() for v in (prediction, target, state)):
        raise ValueError("Invalid validation values or first-action padding")
    error = np.abs((prediction - target) * SCALE[:action_dim])
    baseline = np.abs((state.reshape(7)[:action_dim] - target[0]) * SCALE[:action_dim])
    valid = ~padding
    return np.array([1., valid.sum(), error[valid, :6].sum(), error[valid, 6].sum() if action_dim == 7 else 0.,
                     error[0, :6].sum(), error[0, 6] if action_dim == 7 else 0., baseline[:6].sum(),
                     baseline[6] if action_dim == 7 else 0.], dtype=np.float64)


def summarize_joint_metrics(totals, *, action_dim=7):
    schema_for(action_dim)
    n, action_n = max(float(totals[0]), 1.), max(float(totals[1]), 1.)
    result = dict(anchors=int(totals[0]), valid_actions=int(totals[1]),
                  joint_mae_deg=float(totals[2] / (6 * action_n)), gripper_mae_cm=float(totals[3] / action_n),
                  first_joint_mae_deg=float(totals[4] / (6 * n)), first_gripper_mae_cm=float(totals[5] / n),
                  state_copy_baseline_first_joint_mae_deg=float(totals[6] / (6 * n)),
                  state_copy_baseline_first_gripper_mae_cm=float(totals[7] / n))
    result["selection_score"] = result["first_joint_mae_deg"]
    if action_dim == 7:
        result["selection_score"] += GRIPPER_SCORE_WEIGHT * result["first_gripper_mae_cm"]
    else:
        for key in ("gripper_mae_cm", "first_gripper_mae_cm", "state_copy_baseline_first_gripper_mae_cm"):
            del result[key]
    return result


@torch.no_grad()
def evaluate(model, dataset, anchors, rank, world, device, seed, step, output):
    model.eval()
    totals = np.zeros(12, dtype=np.float64)
    started = time.monotonic()
    if rank == 0:
        print(json.dumps(dict(event="validation_started", step=step, anchors=len(anchors))), flush=True)
    for ordinal in range(rank, len(anchors), world):
        episode, frame = anchors[ordinal]
        example = dataset.get_frame(episode, frame)
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(seed + episode * 100003 + frame)
            torch.cuda.manual_seed(seed + episode * 100003 + frame)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = model.predict_action([example["image"]], [example["lang"]],
                    state=np.stack([example["state"]]))["normalized_actions"][0]
                losses = model([example])
        values = joint_metric_sums(predicted, example["action"], example["state"], example["action_is_pad"], action_dim=dataset.action_dim)
        future = (~example["video_is_pad"].reshape(-1, model.vj_encoder.config.tubelet_size).any(-1))[1:].sum()
        totals[:8] += values
        totals[8:] += [float(losses["action_loss"]) * values[1], values[1], float(losses["wm_loss"]) * future, future]
    if not np.isfinite(totals).all():
        raise RuntimeError("Nonfinite validation values")
    reduced = torch.as_tensor(totals, device=device)
    dist.all_reduce(reduced)
    totals = reduced.cpu().numpy()
    metrics = summarize_joint_metrics(totals, action_dim=dataset.action_dim)
    if metrics["anchors"] != len(anchors):
        raise RuntimeError("Validation skipped or repeated anchors")
    metrics.update(flow_loss=float(totals[8] / max(totals[9], 1)),
                   wm_loss_weighted_0_1=float(totals[10] / max(totals[11], 1)))
    result = dict(step=step, evaluation_seed=seed, seconds=time.monotonic() - started,
                  overall=metrics, group=dataset.group, selection=selection_description(dataset.action_dim),
                  interpretation="offline held-out action prediction; not simulator task success rate")
    if rank == 0:
        shared.write_json(output / f"validation_{step:06d}.json", result)
        print(json.dumps(dict(event="validation", step=step, **metrics)), flush=True)
    return result


def _replace_alias(root, alias, destination):
    temporary = root / (alias + ".tmp")
    if temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(destination.name, target_is_directory=True)
    os.replace(temporary, root / alias)


def publish_best(root, source, metadata):
    """Idempotently finish best publication, including recovery after interruption."""
    name = f"best_step_{metadata['step']:06d}"
    destination = root / name
    if not destination.exists():
        staging = Path(tempfile.mkdtemp(prefix=name + ".tmp_", dir=root))
        os.link(source / "model.pt", staging / "model.pt")
        best_metadata = dict(metadata, schema=metadata.get("schema", SCHEMA), kind="model_only_best", complete=True,
                             model_bytes=(staging / "model.pt").stat().st_size)
        best_metadata.pop("training_bytes", None)
        shared.write_json(staging / "manifest.json", best_metadata)
        os.replace(staging, destination)
    else:
        saved = json.loads((destination / "manifest.json").read_text())
        if saved.get("step") != metadata["step"] or saved.get("best_score") != metadata.get("best_score"):
            raise ValueError("Existing best generation disagrees with resumable checkpoint")
        if not os.path.samefile(source / "model.pt", destination / "model.pt"):
            raise ValueError("Interrupted best publication does not reference the last model")
    _replace_alias(root, "best", destination)


def reconcile_best_checkpoint(root, last, manifest):
    """A crash between last/best alias replacement must not lose the best model."""
    if manifest.get("checkpoint_improved_best"):
        publish_best(root, last, manifest)
    else:
        best_manifest = root / "best/manifest.json"
        if not best_manifest.exists():
            raise ValueError("A non-improving last checkpoint requires its earlier best model")
        best = json.loads(best_manifest.read_text())
        if best.get("best_score") != manifest.get("best_score") or best["step"] > manifest["step"]:
            raise ValueError("Best checkpoint disagrees with the resumable training cursor")


def write_checkpoint_files(root, model_state, training_state, metadata, is_best):
    """Publish complete generations atomically and retain only this run's two aliases."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = f"last_step_{metadata['step']:06d}"
    staging = Path(tempfile.mkdtemp(prefix=name + ".tmp_", dir=root))
    torch.save(model_state, staging / "model.pt")
    torch.save(training_state, staging / "training.pt")
    metadata = dict(metadata, checkpoint_improved_best=bool(is_best))
    shared.write_json(staging / "manifest.json", dict(metadata, schema=metadata.get("schema", SCHEMA), kind="resumable_last", complete=True,
                                                     model_bytes=(staging / "model.pt").stat().st_size,
                                                     training_bytes=(staging / "training.pt").stat().st_size))
    destination = root / name
    if destination.exists():
        raise FileExistsError(f"Checkpoint generation already exists: {destination}")
    os.replace(staging, destination)
    _replace_alias(root, "last", destination)
    if is_best:
        publish_best(root, destination, metadata)
    keep = {(root / alias).resolve() for alias in ("best", "last") if (root / alias).exists()}
    for pattern in ("last_step_*", "best_step_*"):
        for path in root.glob(pattern):
            if path.is_dir() and ".tmp_" not in path.name and path.resolve() not in keep:
                shutil.rmtree(path)


def _publish_model_only_generation(root, alias, model_state, metadata, *, link_source=None):
    """Replace one alias, then release its old payload before the next save."""
    destination = root / f"{alias}_step_{metadata['step']:06d}"
    if destination.exists():
        raise FileExistsError(f"Checkpoint generation already exists: {destination}")
    previous = (root / alias).resolve() if (root / alias).exists() else None
    if previous is not None and (previous.parent != root or not previous.name.startswith(alias + "_step_")):
        raise ValueError("Refusing to replace a checkpoint outside this run's generated aliases")
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp_", dir=root))
    published = False
    try:
        if link_source is None:
            torch.save(model_state, staging / "model.pt")
        else:
            os.link(link_source / "model.pt", staging / "model.pt")
        shared.write_json(staging / "manifest.json", dict(metadata,
            schema=metadata.get("schema", SCHEMA), kind=f"model_only_{alias}", complete=True,
            checkpoint_policy=MODEL_ONLY_CHECKPOINT_POLICY,
            optimizer_retained=False, exact_resume_supported=False, training_bytes=0,
            model_bytes=(staging / "model.pt").stat().st_size))
        os.replace(staging, destination)
        _replace_alias(root, alias, destination)
        published = True
    finally:
        # An unsuccessful write must leave the prior complete alias usable.
        if staging.exists():
            shutil.rmtree(staging)
        if not published and destination.exists() and (root / alias).resolve() != destination:
            shutil.rmtree(destination)
        pending_alias = root / (alias + ".tmp")
        if not published and pending_alias.is_symlink() and pending_alias.resolve() == destination:
            pending_alias.unlink()
    if previous is not None and previous != destination:
        other_alias = root / ("last" if alias == "best" else "best")
        if not other_alias.exists() or other_alias.resolve() != previous:
            shutil.rmtree(previous)
    return destination


def write_model_only_checkpoint_files(root, model_state, metadata, *, is_best, final):
    """One best during training; one final last. At most two model payloads.

    Updating best temporarily holds old best plus its replacement, then deletes
    old best. A final improvement hardlinks last to the new best, so it never
    introduces a third payload. No optimizer, scheduler or RNG is serialized.
    """
    if not is_best and not final:
        return
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "last").exists():
        raise ValueError("Model-only training is already finalized; use a fresh run directory")
    metadata = dict(metadata, checkpoint_improved_best=bool(is_best), final=bool(final))
    best = root / "best"
    if not is_best and not best.exists():
        raise ValueError("A non-improving final checkpoint requires an earlier best model")
    if is_best:
        best = _publish_model_only_generation(root, "best", model_state, metadata)
    if final:
        _publish_model_only_generation(root, "last", model_state, metadata,
                                       link_source=best if is_best else None)


def save_checkpoint(model, optimizer, scheduler, output, progress, is_best, rank, world, provenance,
                    *, checkpoint_policy=RESUMABLE_CHECKPOINT_POLICY, final=False):
    if checkpoint_policy == MODEL_ONLY_CHECKPOINT_POLICY:
        if not is_best and not final:
            return
        if rank == 0:
            started = time.monotonic()
            write_model_only_checkpoint_files(output / "checkpoints", model.state_dict(),
                dict(world_size=world, **progress, **provenance), is_best=is_best, final=final)
            print(json.dumps(dict(event="checkpoint_saved", step=progress["step"], best=is_best,
                final=final, checkpoint_policy=checkpoint_policy,
                seconds=time.monotonic() - started)), flush=True)
        dist.barrier()
        return
    if checkpoint_policy != RESUMABLE_CHECKPOINT_POLICY:
        raise ValueError(f"Unknown checkpoint policy: {checkpoint_policy}")
    rngs = [None] * world
    dist.all_gather_object(rngs, shared.get_rng())
    if rank == 0:
        started = time.monotonic()
        training = dict(provenance, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng_by_rank=rngs,
                        world_size=world, schema=provenance.get("schema", SCHEMA), **progress)
        write_checkpoint_files(output / "checkpoints", model.state_dict(), training,
                               dict(world_size=world, **progress, **provenance), is_best)
        print(json.dumps(dict(event="checkpoint_saved", step=progress["step"], best=is_best,
                              seconds=time.monotonic() - started)), flush=True)
    dist.barrier()


def build_contract(cfg, normalization, split_manifest=None):
    contract, _ = shared.training_contract(cfg)
    policy = configured_checkpoint_policy(cfg)
    if policy != RESUMABLE_CHECKPOINT_POLICY:
        contract.update(checkpoint_policy=policy)
    action_dim = int(cfg["framework"]["action_model"]["action_dim"])
    counts = dict(split_manifest["counts"]) if split_manifest is not None else {"train": 90, "val": 10}
    if counts not in ({"train": 90, "val": 10}, {"train": 900, "val": 100}):
        raise ValueError("Expected the common 90/10 or 900/100 episode split")
    contract.update(schema=schema_for(action_dim), normalization=normalization,
                    validation_selection_weight=GRIPPER_SCORE_WEIGHT if action_dim == 7 else 0.,
                    split_rule=f"episode permutation seed42, first{counts['val']} validation, remaining{counts['train']} train",
                    initialization=f"official pretrained, four joint{action_dim} I/O tensors reset")
    # Leave the historical 100-episode contract byte-for-byte compatible.
    if counts["train"] == 900:
        contract.update(source_episode_count=1000, split_counts=counts,
                        validation_anchor_count=500, excluded_evaluation_seed_range=[10000, 10099],
                        validation_split_note="Recompute seed42 permutation over all 1000 episode IDs; the old 100-episode holdout is not preserved. Physical evaluation seeds10000..10099 remain excluded.")
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return contract, fingerprint


def validate_anchor_count(dataset, anchors):
    """Five fixed anchors from every held-out episode, for both supported sizes."""
    expected = 5 * dataset.split_manifest["counts"]["val"]
    if len(anchors) != expected:
        raise ValueError(f"Expected exactly {expected} fixed validation anchors (all validation episodes x 5)")
    return expected


def action_schema(dataset, provenance, *, schema_name=None):
    dimension = dataset.action_dim
    result = dict(provenance, schema=schema_name or schema_for(dimension), state_dim=7, action_dim=dimension,
        action_horizon=7, video_horizon=8, fps=30,
        representation="absolute joint targets; joint1..6 + full_gripper_width" if dimension == 7 else "absolute arm joint targets; joint1..6 only",
        dataset_units=["deg"] * 6 + ["cm"], normalization=dataset.normalization,
        cameras=["up", "wrist"])
    if dimension == 6:
        result.update(state_units=["deg"] * 6 + ["cm"], action_units=["deg"] * 6,
            fixed_gripper_full_width_m=dataset.normalization["fixed_gripper_full_width_m"],
            gripper_state="observed measured full width; never replaced by the fixed command",
            gripper_action="excluded from model output, diffusion noise, training target and loss")
    return result


def setup_distributed():
    rank, world, local = (int(os.environ.get(key, default)) for key, default in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    extra = {}
    if "RANK" not in os.environ:
        fd, path = tempfile.mkstemp(prefix="table-buttons-ddp-")
        os.close(fd)
        os.unlink(path)
        extra = dict(init_method="file://" + path, rank=0, world_size=1)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=60), **extra)
    return rank, world, local, device


def main(*, dataset_class=None, evaluator=evaluate, contract_builder=build_contract,
         anchor_validator=validate_anchor_count, schema_name=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", type=Path, help="Resume only this run's own checkpoints/last with matching state/action schema")
    parser.add_argument("--probe-steps", type=int, default=0, help="Resource probe: run N optimizer updates, no checkpoints or validation")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    if args.probe_steps < 0 or (args.probe_steps and args.resume):
        raise ValueError("Probe requires positive steps and fresh initialization")
    from omegaconf import OmegaConf
    from pressb.training.image_transforms import build_image_transforms, normalize_image_transforms_config

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.set))
    cfg.image_transforms = normalize_image_transforms_config(cfg.get("image_transforms"))
    if args.run_id:
        cfg.run_id = args.run_id
    checkpoint_policy = configured_checkpoint_policy(cfg, args.resume)
    warm_settings = warm_start_configuration(cfg, args.resume)
    action_dim = int(cfg.framework.action_model.action_dim)
    schema = schema_name or schema_for(action_dim)
    if warm_settings is not None and schema != "vlajepa.joint7.v1":
        raise ValueError("Model-only warm start requires the joint7_generic training profile")
    schema_for(action_dim)  # Validate action dimensions for injected data profiles too.
    if cfg.framework.action_model.state_dim != 7:
        raise ValueError("This trainer retains seven measured state coordinates")
    if int(cfg.seed) != 42:
        raise ValueError("These nine runs use the fixed common episode split seed42")
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
    from starVLA.dataloader.table_buttons_lerobot import TableButtonsLeRobotDataset, collate_fn
    if dataset_class is None:
        dataset_class = TableButtonsLeRobotDataset
        if action_dim == 6:
            from starVLA.dataloader.table_buttons_joint6_lerobot import TableButtonsJoint6LeRobotDataset
            dataset_class = TableButtonsJoint6LeRobotDataset
    common = dict(root=cfg.datasets.vla_data.data_root_dir, split_manifest=output / "split.json", action_horizon=7,
                  video_horizon=8, resolution=int(cfg.datasets.vla_data.resolution_size), seed=int(cfg.seed),
                  frame_cache_size=int(cfg.datasets.vla_data.frame_cache_size), decoder_cache_size=int(cfg.datasets.vla_data.decoder_cache_size))
    train_data = dataset_class(split="train", sample_stride=int(cfg.datasets.vla_data.sample_stride),
                              image_transforms=build_image_transforms(cfg.get("image_transforms"), split="train", seed=int(cfg.seed)), **common)
    val_data = dataset_class(split="val",
                            image_transforms=build_image_transforms(cfg.get("image_transforms"), split="val", seed=int(cfg.seed)), **common)
    anchors = shared.validation_anchors(val_data, int(cfg.trainer.val_anchors_per_episode), int(cfg.trainer.val_max_samples))
    anchor_validator(val_data, anchors)
    batch_size, accumulation = int(cfg.datasets.vla_data.per_device_batch_size), int(cfg.trainer.gradient_accumulation_steps)
    geometry = shared.epoch_geometry(len(train_data), world, batch_size, accumulation, int(cfg.trainer.target_epochs))
    cfg.trainer.max_train_steps = geometry["target_steps"]
    cfg.trainer.num_warmup_steps = max(1, round(.05 * geometry["target_steps"]))
    maximum = args.probe_steps or geometry["target_steps"]
    contract, fingerprint = contract_builder(OmegaConf.to_container(cfg, resolve=True), train_data.normalization,
                                           train_data.split_manifest)
    provenance = dict(dataset_sha256=train_data.export_manifest_sha256, split_sha256=train_data.split_manifest_sha256,
                      dataset_fingerprint_sha256=train_data.fingerprint_sha256, training_fingerprint=fingerprint,
                      dataset_group=train_data.group)
    if schema_name is not None:
        provenance.update(schema=schema, state_dim=7, action_dim=action_dim,
                          task_id=train_data.task_id, instructions=[train_data.instruction])
    if checkpoint_policy == MODEL_ONLY_CHECKPOINT_POLICY:
        provenance.update(checkpoint_policy=checkpoint_policy, optimizer_retained=False,
                          exact_resume_supported=False)
    if action_dim == 6:
        provenance.update(schema=schema, state_dim=7, action_dim=6,
            fixed_gripper_full_width_m=train_data.normalization["fixed_gripper_full_width_m"])
    warm_start = inspect_warm_start(cfg, schema=schema, provenance=provenance,
                                   normalization=train_data.normalization, output=output) if warm_settings else None
    if warm_start is not None:
        provenance.update(warm_start_source_step=warm_start["source_step"],
                          warm_start_checkpoint_sha256=warm_start["checkpoint_sha256"],
                          warm_start_source_training_fingerprint=warm_start["source_training_fingerprint"],
                          optimizer_reinitialized=True, scheduler_reinitialized=True)
    progress = dict(step=0, epoch=0, batch_in_epoch=0, best_score=None)
    training, checkpoint_dir = None, None
    if args.resume:
        checkpoint_dir = args.resume.resolve(strict=True)
        if checkpoint_dir.parent != (output / "checkpoints").resolve():
            raise ValueError("Resume is restricted to this run's own checkpoint directory")
        manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
        if manifest.get("schema") != schema or manifest.get("kind") != "resumable_last" or not manifest.get("complete"):
            raise ValueError("Resume requires a complete last checkpoint with matching joint action schema")
        for key, value in provenance.items():
            if manifest.get(key) != value:
                raise ValueError(f"Resume contract changed: {key}")
        for name in ("model", "training"):
            if (checkpoint_dir / f"{name}.pt").stat().st_size != manifest[f"{name}_bytes"]:
                raise ValueError(f"Incomplete {name} checkpoint")
        training = torch.load(checkpoint_dir / "training.pt", map_location="cpu", weights_only=False, mmap=True)
        if training.get("schema") != schema or training["world_size"] != world:
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
        loading = dict(initialization=f"resume_this_joint{action_dim}_run", checkpoint=str(checkpoint_dir), source_step=progress["step"])
    elif warm_start is not None:
        loading = load_model_only_warm_start(model, warm_start)
    else:
        checkpoint_sha = cfg.trainer.get("pretrained_sha256")
        if action_dim == 6:
            verified_sha = shared.sha256(cfg.trainer.pretrained_checkpoint)
            if checkpoint_sha and checkpoint_sha != verified_sha:
                raise ValueError("Official base checkpoint checksum differs from configured source")
            checkpoint_sha = verified_sha
        loading = load_joint_pretrained(model, cfg.trainer.pretrained_checkpoint, action_dim=action_dim)
        loading["checkpoint_path"] = str(Path(cfg.trainer.pretrained_checkpoint).resolve())
        loading["checkpoint_sha256"] = checkpoint_sha or shared.sha256(cfg.trainer.pretrained_checkpoint)
        if action_dim == 6:
            loading["checkpoint_digest_verified_this_run"] = True
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
        loading_name = "resume_loading.json" if args.resume else "warm_start_loading.json" if warm_start is not None else "pretrained_loading.json"
        shared.write_json(output / loading_name, loading)
        if warm_start is not None:
            shared.write_json(output / "warm_start.json", dict(**warm_start, target_epochs=int(cfg.trainer.target_epochs),
                additional_target_steps=geometry["target_steps"], cumulative_target_steps=warm_start["source_step"]+geometry["target_steps"]))
        shared.write_json(output / "parameters.json", parameter_report)
        shared.write_json(output / "action_schema.json", action_schema(train_data, provenance, schema_name=schema_name))
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
        evaluator(model, val_data, anchors, rank, world, device, int(cfg.seed) + 500000, 0, output)
    if rank == 0:
        shared.write_json(output / "initialization.json", dict(step=progress["step"], **geometry,
            global_batch_size=world * batch_size * accumulation, probe=bool(args.probe_steps), **provenance))
    optimizer.zero_grad(set_to_none=True)
    started, checked_gradients = time.monotonic(), False
    while progress["step"] < maximum:
        if train_data.image_transforms is not None:
            train_data.image_transforms.set_epoch(progress["epoch"])
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
            probe_batch_colors = {}
            if args.probe_steps:
                for batch in batches:
                    for example in batch:
                        color = example.get("target_color", "unspecified")
                        probe_batch_colors[color] = probe_batch_colors.get(color, 0) + 1
            if args.probe_steps and progress["step"] == 0 and rank == 0:
                examples = [example for batch in batches for example in batch]
                shared.write_json(output / "probe_batch_contract.json", dict(
                    batch_size=len(examples), target_color_counts=probe_batch_colors,
                    instructions=sorted({example["lang"] for example in examples}),
                    measured_state_shape=list(np.stack([row["state"] for row in batches[0]]).shape),
                    action_target_shape=list(np.stack([row["action"] for row in batches[0]]).shape),
                    action_head_dimension=model.action_model.action_dim,
                    action_head_horizon=model.action_model.action_horizon,
                    interfaces=interface_evidence(model), normalization=train_data.normalization))
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
            if args.probe_steps:
                record["batch_color_counts"] = probe_batch_colors
            if rank == 0:
                with (output / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                shared.write_json(output / "status.json", dict(status="training", target_steps=maximum,
                                  group=train_data.group, probe=bool(args.probe_steps), **record))
                if progress["step"] == 1 or progress["step"] % int(cfg.trainer.logging_frequency) == 0 or args.probe_steps:
                    print(json.dumps(record), flush=True)
            if not args.probe_steps and (epoch_finished or progress["step"] % int(cfg.trainer.eval_interval) == 0 or progress["step"] == maximum):
                metrics = evaluator(model, val_data, anchors, rank, world, device, int(cfg.seed) + 500000, progress["step"], output)
                score = metrics["overall"]["selection_score"]
                is_best = progress["best_score"] is None or score < progress["best_score"]
                if is_best:
                    progress["best_score"] = score
                save_checkpoint(model, optimizer, scheduler, output, progress, is_best, rank, world, provenance,
                                checkpoint_policy=checkpoint_policy, final=progress["step"] == maximum)
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
            model.eval()
            sample = batches[0][0]
            with torch.inference_mode(), torch.random.fork_rng(devices=[device.index]):
                torch.manual_seed(int(cfg.seed) + 900000)
                torch.cuda.manual_seed(int(cfg.seed) + 900000)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    generated = model.predict_action([sample["image"]], [sample["lang"]],
                        state=np.stack([sample["state"]]))["normalized_actions"]
            if generated.shape != (1, 7, action_dim) or not np.isfinite(generated).all():
                raise RuntimeError("Probe generated the wrong action shape or nonfinite actions")
            completed.update(generated_action_shape=list(generated.shape),
                             generated_action_finite=True, generated_actions=generated.tolist())
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
