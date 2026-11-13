#!/usr/bin/env python3
"""Archive completed H200 π0.5 runs and evaluate them serially in local Isaac."""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import inspect
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from run_cube_stack_sft_eval import (
    ROOT, Evaluation, gpu_snapshot, port_available, read_json, sha256, summarize, write_json,
)
from run_manipulation_sft_queue import SSH_OPTIONS, digest, fingerprint, tree_manifest

TASKS = ("cube_stack", "tabletop_cleanup")
DEFAULT_DATASETS = Path("/home/pengguanqi/Datasets")
MODEL_PYTHON = ROOT / ".conda/envs/pressb/bin/python"
RUN_METADATA = ("metrics.jsonl", "resolved_config.json", "split.json")
TASK_LIMITS = {"cube_stack": 45, "tabletop_cleanup": 120}
TASK_SEEDS = {"cube_stack": 2026111101, "tabletop_cleanup": 2026111102}
DEFAULT_DATASET_SUFFIX = "500_20261110"


class TransferUnavailable(RuntimeError):
    pass


def remote_json(host, source, *args):
    command = shlex.join(["python3", "-", *map(str, args)])
    result = subprocess.run(["ssh", *SSH_OPTIONS, host, command], input=source, text=True,
                            capture_output=True, timeout=180)
    if result.returncode == 255:
        raise TransferUnavailable(result.stderr.strip() or "SSH connection unavailable")
    result.check_returncode()
    return json.loads(result.stdout)


def remote_receipts(host, root):
    return remote_json(host, """
import json, sys
from pathlib import Path
root = Path(sys.argv[1]) / "receipts"
print(json.dumps({task: json.loads(path.read_text()) for task in sys.argv[2:]
                  if (path := root / f"{task}.json").is_file()}))
""", root, *TASKS)


def inventory(host, root, task, receipt):
    expected = root / "outputs" / task
    if (receipt.get("task") != task or receipt.get("status") != "completed"
            or receipt.get("exit_code") != 0 or receipt.get("run") != str(expected)
            or not isinstance(receipt.get("checkpoint"), str)):
        raise ValueError(f"{task} has an invalid completion receipt")
    source = "import hashlib, json, math, os, sys\nfrom pathlib import Path\n"
    source += inspect.getsource(digest) + inspect.getsource(tree_manifest)
    source += """
run = Path(sys.argv[1]).resolve(strict=True)
checkpoint = Path(sys.argv[2]).resolve(strict=True)
if checkpoint != (run / "checkpoints/best/pretrained_model").resolve(strict=True):
    raise ValueError("Completion receipt does not refer to the best validation checkpoint")
if not checkpoint.is_relative_to(run / "checkpoints"):
    raise ValueError("Checkpoint escapes the completed run")
metadata = {name: {"bytes": (run / name).stat().st_size, "sha256": digest(run / name)}
            for name in sys.argv[3:]}
manifest = tree_manifest(checkpoint)
files = {f"pretrained_model/{name}": value for name, value in manifest["files"].items()}
links = {f"pretrained_model/{name}": value for name, value in manifest["symlinks"].items()}
print(json.dumps({"run": str(run), "checkpoint": str(checkpoint),
                  "manifest": {"files": {**files, **metadata}, "symlinks": links}}))
"""
    return remote_json(host, source, expected, receipt["checkpoint"], *RUN_METADATA)


def rsync(host, sources, dest):
    command = ["rsync", "-a", "--partial", "--protect-args", "-e", shlex.join(["ssh", *SSH_OPTIONS]),
               "--", *[f"{host}:{source}" for source in sources], str(dest)]
    result = subprocess.run(command, timeout=1800)
    if result.returncode in (10, 12, 20, 23, 24, 30, 35, 255):
        raise TransferUnavailable(f"rsync exited with status {result.returncode}")
    result.check_returncode()


def archive_task(args, task, receipt):
    expected = inventory(args.host, args.remote_root, task, receipt)
    destination = args.archive_root / task
    staging = args.archive_root / f".{task}.incoming"
    if not destination.exists():
        (staging / "pretrained_model").mkdir(parents=True, exist_ok=True)
        rsync(args.host, [expected["checkpoint"] + "/"], staging / "pretrained_model")
        rsync(args.host, [f"{expected['run']}/{name}" for name in RUN_METADATA], staging)
        if tree_manifest(staging) != expected["manifest"]:
            raise ValueError(f"Transferred {task} checkpoint and logs differ from H200 files")
        staging.rename(destination)
    elif tree_manifest(destination) != expected["manifest"]:
        raise ValueError(f"Existing archive differs from completed H200 run: {destination}")
    archive = dict(task=task, archive_path=str(destination), manifest=expected["manifest"],
                   manifest_sha256=fingerprint(expected["manifest"]), verified_unix=time.time(),
                   completion_receipt=receipt)
    write_json(args.queue_root / task / "archive.json", archive)
    return archive


def inspect_archive(task, archive):
    manifest = archive["manifest"]
    if (archive.get("task") != task or not archive.get("verified_unix")
            or archive.get("manifest_sha256") != fingerprint(manifest)):
        raise ValueError(f"Invalid {task} archive receipt")
    root = Path(archive["archive_path"]).resolve(strict=True)
    if tree_manifest(root) != manifest:
        raise ValueError(f"Archived {task} files differ from their verified receipt")
    return inspect_checkpoint(root / "pretrained_model",
                              model_sha=manifest["files"]["pretrained_model/model.safetensors"]["sha256"])


def inspect_checkpoint(path, *, model_sha=None):
    checkpoint = path.resolve(strict=True)
    native = read_json(checkpoint / "config.json")
    sidecar = read_json(checkpoint / "pressb_pi05.json")
    if (native.get("type") != "pi05" or native.get("train_expert_only") is not True
            or type(native.get("chunk_size")) is not int or native["chunk_size"] < 1
            or sidecar.get("format") != "pressb.pi05.v1"
            or sidecar["observation"]["cameras"] != ["up", "wrist"]
            or sidecar["action"]["representation"] != "joint_position"
            or sidecar["action"]["control_hz"] != 30
            or sidecar["action"]["units"] != ["rad"] * 6 + ["m"]
            or type(sidecar["checkpoint_step"]) is not int or sidecar["checkpoint_step"] < 1):
        raise ValueError("Expected an expert-only π0.5 checkpoint for PiPER joint7 at 30 Hz")
    return dict(checkpoint=str(checkpoint), checkpoint_step=sidecar["checkpoint_step"],
                chunk_size=native["chunk_size"], model_sha256=model_sha or sha256(checkpoint / "model.safetensors"))


def dataset_manifest(args, task=None):
    """Resolve one explicitly selected export without changing its task settings."""
    task = task or args.task
    if args.dataset is not None:
        dataset = Path(str(args.dataset).replace("{task}", task)).expanduser()
    elif args.dataset_name is not None:
        dataset = args.dataset_root / args.dataset_name.replace("{task}", task)
    else:
        suffix = args.dataset_suffix or DEFAULT_DATASET_SUFFIX
        dataset = args.dataset_root / f"pressb_{task}_{suffix}"
    dataset = dataset.expanduser().resolve()
    return dataset if dataset.is_file() or dataset.suffix == ".json" else dataset / "meta/export_manifest.json"


def merge_task_config(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        result[key] = (merge_task_config(result[key], value)
                       if isinstance(result.get(key), dict) and isinstance(value, dict) else deepcopy(value))
    return result


def evaluation_request(args, task=None):
    manifest = dataset_manifest(args, task)
    override = args.task_config.expanduser().resolve() if args.task_config else None
    return dict(dataset_manifest=str(manifest),
                dataset_manifest_sha256=sha256(manifest) if manifest.is_file() else None,
                task_config=str(override) if override else None,
                task_config_sha256=sha256(override) if override and override.is_file() else None,
                performance_profile=args.performance_profile, episodes=args.episodes,
                num_envs=args.num_envs, batch_size=args.batch_size,
                simulation_workers=args.simulation_workers,
                total_envs=args.num_envs * args.simulation_workers)


def prepare_config(args, checkpoint):
    from importlib import import_module

    from pressb.experiments.config import load_experiment
    from pressb.runtime.trainer import TrainerConfig

    manifest = dataset_manifest(args)
    export = read_json(manifest)
    if not isinstance(export, dict):
        raise ValueError(f"Missing or invalid dataset export manifest: {manifest}")
    if export.get("task_id") != args.task or export.get("fps") != 30:
        raise ValueError("Dataset manifest differs from the requested evaluation task")
    template = ROOT / "configs/experiments/cube_stack_pi05_eval.json"
    config = deepcopy(load_experiment(template)["experiment"])
    config.update(name=f"{args.task}_pi05_sft_eval",
                  description="Expert-only π0.5 evaluation on independent sampled layouts.")
    task = deepcopy(export["raw_manifest"]["config"])
    task.update(horizon=checkpoint["chunk_size"], num_envs=args.num_envs, gpu=args.sim_gpu,
                terminate_on_success=True, max_episode_seconds=TASK_LIMITS[args.task],
                log_dir=str(args.output / "simulation"))
    if args.task_config:
        overrides = read_json(args.task_config.expanduser())
        if not isinstance(overrides, dict):
            raise ValueError("--task-config must name a JSON object of task configuration overrides")
        # Runtime allocation and the model's action shape remain launcher-owned.
        if set(overrides) & {"horizon", "num_envs", "gpu", "log_dir", "terminate_on_success"}:
            raise ValueError("--task-config cannot override horizon, num_envs, gpu, log_dir or terminate_on_success")
        task = merge_task_config(task, overrides)
    scene = Path(task["scene_config"].replace("${project_root}", str(ROOT))).expanduser()
    task["scene_config"] = str(ROOT / scene)
    if args.performance_profile == "fast":
        task["color_response_workers"] = 4
    task_config = import_module(f"pressb.tasks.{args.task}.configuration").load_config(deepcopy(task))
    if task_config["control_hz"] != 30:
        raise ValueError("Simulation control rate differs from the checkpoint")
    config["task"] = dict(plugin=args.task, config=task)
    config["model"]["config"].update(checkpoint=checkpoint["checkpoint"], device=f"cuda:{args.inference_gpu}")
    config["nodes"]["inference"]["service"]["batch_size"] = args.batch_size
    if args.performance_profile == "fast":
        config["nodes"]["inference"]["service"]["collect_timing"] = True
    config["nodes"]["inference"]["server"]["port"] = args.inference_port
    config["nodes"]["simulation"]["server"]["port"] = args.simulation_port
    config["run"].update(seed=TASK_SEEDS[args.task], output_dir=str(args.output / "run"))
    config["run"]["rollout"].update(eval_episodes=args.episodes, num_envs=args.num_envs,
                                   execution_horizon=min(7, task["horizon"]),
                                   max_transitions=max(100000, math.ceil(args.episodes * task["max_episode_seconds"] * 30)))
    if args.performance_profile == "fast":
        config["run"]["rollout"].update(terminal_observation_mode="state_only_if_no_bootstrap",
                                       fused_step_reset=True)
    path = args.output / "experiment.json"
    write_json(path, config)
    compiled = load_experiment(path)
    TrainerConfig(**deepcopy(compiled["nodes"]["trainer"]))
    return path, compiled


def summarize_single(run, episodes):
    result = summarize(run, episodes)
    events = run / "events.jsonl"
    errors = [row for line in events.read_text().splitlines() if line.strip()
              if (row := json.loads(line)).get("event") in
              ("context_cleanup_failed", "simulation_close_failed", "observer_close_failed")] if events.is_file() else []
    cleanup_error = result["runner_status"].get("context_cleanup_error")
    if cleanup_error or errors:
        raise RuntimeError(f"Evaluation cleanup failed: {errors or cleanup_error}")
    return result


class Pi05Evaluation(Evaluation):
    def __init__(self, args):
        super().__init__(args)
        self.state["mode"] = "formal" if args.episodes == 100 else "diagnostic"
        self.state["task"] = args.task
        self.state["num_envs_per_worker"] = args.num_envs
        self.state["simulation_workers"] = args.simulation_workers
        self.state["total_envs"] = args.num_envs * args.simulation_workers

    def sharded_arguments(self, path, checkpoint):
        args = self.args
        command = ["--config", str(path), "--output", str(args.output / "sharded"),
                   "--workers", str(args.simulation_workers), "--seed", str(TASK_SEEDS[args.task]),
                   "--expected-task-id", args.task,
                   "--expected-weights-sha256", checkpoint["model_sha256"]]
        for name in ("num_envs", "episodes", "sim_gpu", "inference_gpu", "batch_size",
                     "simulation_port", "inference_port", "sim_python", "model_python",
                     "startup_timeout", "poll_seconds"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        if args.validate_only:
            command.append("--validate-only")
        return command

    def run_sharded(self, path, checkpoint):
        from run_sharded_evaluation import main as sharded_main

        self.update("preparing_workers" if self.args.validate_only else "evaluating",
                    sharded_output=str(self.args.output / "sharded"))
        code = sharded_main(self.sharded_arguments(path, checkpoint))
        if code:
            raise RuntimeError(f"Sharded evaluation failed; see {self.args.output / 'sharded/status.json'}")

    def run_single(self, path, compiled, checkpoint):
        args = self.args
        for port in (args.simulation_port, args.inference_port):
            port_available(port)
        self.launch("simulation", args.sim_python, path)
        self.launch("inference", args.model_python, path)
        self.await_health(compiled)
        inference = read_json(args.output / "inference.health.json")
        simulation = read_json(args.output / "simulation.health.json")
        if inference.get("weights_sha256") != checkpoint["model_sha256"] or simulation.get("task_id") != args.task:
            raise ValueError("Live nodes do not match the verified checkpoint and requested task")
        self.launch("eval", args.model_python, path)
        while self.children["eval"].poll() is None:
            self.check_children()
            self.update("evaluating", progress=read_json(args.output / "run/status.json", {}))
            self.pause(args.poll_seconds)
        self.check_children()
        if self.children["eval"].returncode:
            raise RuntimeError(f"Eval runner failed, see {args.output / 'eval.log'}")
        return summarize_single(args.output / "run", args.episodes)

    def run(self):
        args = self.args
        try:
            self.update("verifying_checkpoint")
            checkpoint = (inspect_archive(args.task, read_json(args.archive_receipt))
                          if args.archive_receipt else inspect_checkpoint(args.checkpoint))
            path, compiled = prepare_config(args, checkpoint)
            request = evaluation_request(args)
            self.state.update(checkpoint=checkpoint, seed=TASK_SEEDS[args.task], expected_episodes=args.episodes,
                              evaluation_request=request, experiment_sha256=sha256(path))
            if args.validate_only:
                if args.simulation_workers > 1:
                    self.run_sharded(path, checkpoint)
                self.update("validated", gpu_processes_started=False)
                return 0
            while True:
                hardware = gpu_snapshot()
                if any(gpu not in hardware for gpu in (args.sim_gpu, args.inference_gpu)):
                    raise ValueError("Requested local GPUs are absent")
                if all(hardware[gpu]["free_mib"] >= args.min_free_mib for gpu in (args.sim_gpu, args.inference_gpu)):
                    break
                self.update("waiting_gpus", hardware=hardware)
                self.pause(args.poll_seconds)
            write_json(args.output / "hardware_before.json", hardware)
            if args.simulation_workers > 1:
                self.run_sharded(path, checkpoint)
                result = read_json(args.output / "sharded/result.json")
                episode_file = args.output / "sharded/episodes.jsonl"
            else:
                result = self.run_single(path, compiled, checkpoint)
                episode_file = args.output / "run/episodes.jsonl"
            if args.task == "tabletop_cleanup":
                by_pair = {}
                for line in episode_file.read_text().splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    names = row["info"]["active_names"]
                    if len(names) != 2 or len(set(names)) != 2:
                        raise ValueError("Cleanup evaluation must have exactly two distinct objects")
                    bucket = by_pair.setdefault("+".join(sorted(names)), dict(episodes=0, successes=0))
                    bucket["episodes"] += 1
                    bucket["successes"] += int(row["success"])
                for bucket in by_pair.values():
                    bucket["success_rate"] = bucket["successes"] / bucket["episodes"]
                result["by_object_pair"] = by_pair
            result.update(task=args.task, checkpoint=checkpoint, seed=TASK_SEEDS[args.task],
                          mode=self.state["mode"], experiment_sha256=sha256(path),
                          evaluation_request=request)
            write_json(args.output / "result.json", result)
            self.update("completed", successes=result["successes"], episodes=result["episodes"],
                        success_rate=result["success_rate"])
            return 0
        except BaseException as error:
            self.update("stopped" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                        error_type=type(error).__name__, error=str(error))
            raise
        finally:
            self.cleanup()
            self.state["children_cleaned_up"] = True
            write_json(args.output / "status.json", self.state)


def next_attempt(folder):
    indices = [int(path.name[5:]) for path in folder.glob("eval_*") if path.name[5:].isdigit()]
    return folder / f"eval_{max(indices, default=0)+1:03d}"


def eval_lock(root):
    lock = (root / ".evaluation.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    return lock


def completed_result(task, folder, args=None):
    episodes = args.episodes if args else 100
    mode = "formal" if episodes == 100 else "diagnostic"
    request = evaluation_request(args, task) if args else None
    for path in sorted(folder.glob("eval_*/result.json"), reverse=True):
        state = read_json(path.parent / "status.json", {})
        if state.get("status") != "completed" or not state.get("children_cleaned_up"):
            continue
        result = read_json(path)
        if result.get("task") != task:
            raise ValueError(f"Unexpected completed evaluation result for {task}")
        if result.get("mode") != mode or result.get("episodes") != episodes:
            continue
        if request is not None and result.get("evaluation_request") != request:
            # Receipts written by the old launcher used these fixed defaults.
            old_defaults = (result.get("evaluation_request") is None
                and request["performance_profile"] == "legacy" and request["task_config"] is None
                and request["episodes"] == 100 and request["num_envs"] == 8
                and request["batch_size"] == 4 and request["simulation_workers"] == 1
                and Path(request["dataset_manifest"]).parents[1].name == f"pressb_{task}_{DEFAULT_DATASET_SUFFIX}")
            if not old_defaults:
                continue
        return dict(status="completed", result=result, eval_dir=str(path.parent))
    return None


def queue_command(args, task, archive, output):
    command = [sys.executable, "-u", __file__, "evaluate", "--task", task,
               "--archive-receipt", str(archive), "--output", str(output),
               "--dataset", str(dataset_manifest(args, task))]
    for name in ("episodes", "num_envs", "batch_size", "performance_profile", "simulation_workers",
                 "sim_gpu", "inference_gpu", "sim_python", "model_python", "inference_port",
                 "simulation_port", "min_free_mib", "startup_timeout", "poll_seconds"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    if args.task_config:
        command.extend(["--task-config", str(args.task_config)])
    return command


def eval_queue(args):
    state = {}
    active = None
    active_log = None
    try:
        while True:
            if active and active[2].poll() is not None:
                task, output, worker = active
                result = completed_result(task, args.queue_root / task, args) if worker.returncode == 0 else None
                state[task] = (result if result else
                               dict(status="failed", error=f"Evaluation failed, see {output / 'supervisor.log'}"))
                active_log.close()
                active = active_log = None
            for task in TASKS:
                if state.get(task, {}).get("status") not in ("completed", "failed", "evaluating"):
                    result = completed_result(task, args.queue_root / task, args)
                    if result:
                        state[task] = result
            if all(state.get(task, {}).get("status") in ("completed", "failed") for task in TASKS):
                break
            try:
                receipts = remote_receipts(args.host, args.remote_root)
            except (TransferUnavailable, subprocess.TimeoutExpired) as error:
                receipts = {}
                state["last_network_error"] = str(error)
            for task in TASKS:
                if state.get(task, {}).get("status") in ("completed", "failed", "evaluating"):
                    continue
                folder = args.queue_root / task
                archive = folder / "archive.json"
                receipt = receipts.get(task, {})
                if not archive.exists() and receipt.get("status") in ("failed", "stopped", "cancelled"):
                    state[task] = dict(status="failed", completion_receipt=receipt)
                    continue
                if not archive.exists() and receipt.get("status") == "completed":
                    state[task] = dict(status="archiving")
                    write_json(args.queue_root / "status.json", state)
                    try:
                        archive_task(args, task, receipt)
                    except (TransferUnavailable, subprocess.TimeoutExpired) as error:
                        state[task] = dict(status="archive_retry", error=str(error))
                        continue
                    except (ValueError, OSError, subprocess.CalledProcessError, RuntimeError) as error:
                        state[task] = dict(status="failed", error=str(error))
                        continue
                if not archive.exists():
                    state[task] = dict(status="waiting_training", remote=receipt)
                    continue
                state[task] = dict(status="archived")
                if active is None and (lease := eval_lock(args.queue_root)):
                    with lease:
                        result = completed_result(task, folder, args)
                        if result:
                            state[task] = result
                            continue
                        output = next_attempt(folder)
                        output.mkdir()
                        command = queue_command(args, task, archive, output)
                        active_log = (output / "supervisor.log").open("w")
                        worker = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                                  stdout=active_log, stderr=subprocess.STDOUT,
                                                  pass_fds=(lease.fileno(),))
                    active = task, output, worker
                    state[task] = dict(status="evaluating", eval_dir=str(output), pid=worker.pid)
            write_json(args.queue_root / "status.json", state)
            time.sleep(args.poll_seconds)
        write_json(args.queue_root / "status.json", state)
        if all(state[task]["status"] == "completed" for task in TASKS):
            write_json(args.queue_root / "results.json", {task: state[task]["result"] for task in TASKS})
            return 0
        return 1
    finally:
        if active:
            worker = active[2]
            if worker.poll() is None:
                worker.terminate()
                worker.wait(timeout=60)
            active_log.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    queue = modes.add_parser("queue")
    queue.add_argument("--host", default="h200")
    queue.add_argument("--remote-root", type=Path, required=True)
    queue.add_argument("--queue-root", type=Path, required=True)
    queue.add_argument("--archive-root", type=Path, required=True)
    evaluation = modes.add_parser("evaluate")
    evaluation.add_argument("--task", choices=TASKS, required=True)
    source = evaluation.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive-receipt", type=Path)
    source.add_argument("--checkpoint", type=Path)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--validate-only", "--prepare-config-only", action="store_true",
                            help="Write and validate concrete configs without starting GPU processes")
    for mode in (queue, evaluation):
        mode.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASETS)
        dataset = mode.add_mutually_exclusive_group()
        dataset.add_argument("--dataset", type=Path,
                             help="Dataset directory or export manifest; queue paths must contain {task}")
        dataset.add_argument("--dataset-name", help="Directory under --dataset-root; queue names must contain {task}")
        dataset.add_argument("--dataset-suffix", help=f"Use pressb_<task>_<suffix> (default: {DEFAULT_DATASET_SUFFIX})")
        mode.add_argument("--task-config", type=Path, help="Explicit JSON task overrides merged onto the selected manifest")
        mode.add_argument("--performance-profile", choices=("legacy", "fast"), default="legacy")
        mode.add_argument("--simulation-workers", type=int, choices=(1, 2), default=1,
                          help="Independent simulator processes sharing one inference node")
        mode.add_argument("--episodes", type=int, default=100)
        mode.add_argument("--sim-gpu", type=int, default=0)
        mode.add_argument("--inference-gpu", type=int, default=1)
        mode.add_argument("--sim-python", type=Path, default=MODEL_PYTHON)
        mode.add_argument("--model-python", type=Path, default=MODEL_PYTHON)
        mode.add_argument("--inference-port", type=int, default=18914)
        mode.add_argument("--simulation-port", type=int, default=18915,
                          help="First simulator port; two workers use this and the next port")
        mode.add_argument("--min-free-mib", type=int, default=16000)
        mode.add_argument("--startup-timeout", type=float, default=1200)
        mode.add_argument("--poll-seconds", type=float, default=30)
        mode.add_argument("--num-envs", type=int, default=8,
                          help="Environments per simulator (total = num-envs * simulation-workers)")
        mode.add_argument("--batch-size", type=int, help="Inference batch size (legacy: 4, fast: 8)")
    args = parser.parse_args(argv)
    if args.batch_size is None:
        args.batch_size = 8 if args.performance_profile == "fast" else 4
    if min(args.num_envs, args.batch_size) < 1 or not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("Invalid environment count, batch size or poll interval")
    if args.simulation_workers > args.episodes:
        parser.error("Each simulator worker requires at least one episode")
    if args.performance_profile == "fast" and args.episodes < args.simulation_workers * args.num_envs:
        parser.error("Fast evaluation requires at least one episode per environment across all workers")
    simulation_ports = range(args.simulation_port, args.simulation_port + args.simulation_workers)
    if (args.sim_gpu == args.inference_gpu or min(args.sim_gpu, args.inference_gpu) < 0
            or args.inference_port in simulation_ports or args.episodes < 1
            or min(args.inference_port, args.simulation_port) < 1
            or max(args.inference_port, simulation_ports[-1]) > 65535):
        parser.error("Use distinct nonnegative local GPUs, distinct valid ports, and a positive episode count")
    if args.min_free_mib < 0 or not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error("Use nonnegative free memory and a finite positive startup timeout")
    for name in ("dataset_name", "dataset_suffix"):
        value = getattr(args, name)
        if value is not None and (not value or Path(value).name != value or value in (".", "..")):
            parser.error(f"--{name.replace('_', '-')} must be one directory name")
    if args.mode == "queue":
        selection = args.dataset if args.dataset is not None else args.dataset_name
        if selection is not None and "{task}" not in str(selection):
            parser.error("Queue --dataset/--dataset-name must contain {task}; use --dataset-suffix for both tasks")
    args.dataset_root = args.dataset_root.expanduser().resolve()
    if args.task_config:
        args.task_config = args.task_config.expanduser().resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "queue":
        args.queue_root = args.queue_root.resolve()
        args.archive_root = args.archive_root.expanduser().resolve()
        args.queue_root.mkdir(parents=True, exist_ok=True)
        args.archive_root.mkdir(parents=True, exist_ok=True)
        with (args.queue_root / ".queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for signum in (signal.SIGINT, signal.SIGTERM):
                signal.signal(signum, interrupt)
            return eval_queue(args)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / "status.json").exists():
            raise ValueError("Use a fresh output directory")
        args.smoke_run = None
        runner = Pi05Evaluation(args)
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: setattr(runner, "stopping", True))
        return runner.run()


def interrupt(*_):
    raise InterruptedError("π0.5 queue received SIGINT/SIGTERM")


if __name__ == "__main__":
    raise SystemExit(main())
