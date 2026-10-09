#!/usr/bin/env python3
"""Supervise one fresh 400k residual-on-frozen-noise experiment and 120-task eval.

This program owns only the processes that it starts. Run it in a detached
session or a service for unattended execution. Create OUTPUT/STOP, or send the
supervisor SIGTERM, to drain the learner, save its checkpoint, and skip eval.
Existing experiment directories and model checkpoints are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "outputs/online_rl_fast_20261003"
SERVICE_SOURCE = OLD / "source"
FROZEN_INPUT = ROOT / "outputs/rtx5090_eval_comparison/input"
PYTHON = ROOT / ".conda/envs/pressb/bin/python"
INFERENCE_PYTHON = ROOT / ".conda/envs/vlajepa-inference/bin/python"
PORTS = {"simulation": 19880, "inference": 19891, "learner": 19882}
TRAINING_TRANSITIONS = 400_000
LEARNING_START = 2_000
EVALUATION_EPISODES = 120
EVALUATION_SEED = 20260930
DEFAULT_CONFIG = ROOT / "configs/online_rl_residual_on_noise_400k.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def process_identity(pid):
    directory = Path("/proc") / str(pid)
    stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=pid, argv=(directory / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0"),
                starttime_ticks=int(stat[19]))


def owned(record):
    try:
        return process_identity(record["pid"]) == record
    except (OSError, ValueError):
        return False


class Experiment:
    def __init__(self, output, noise_checkpoint, config=DEFAULT_CONFIG):
        self.output = Path(output).resolve()
        self.noise_checkpoint = Path(noise_checkpoint).resolve()
        self.config_path = Path(config).resolve()
        self.config = None
        self.source = self.output / "source"
        self.children = {}
        self.log_streams = []
        self.stop_requested = False
        self.stop_reason = None
        self.learner = None
        self.state = dict(state="preparing", stage="preflight", supervisor_pid=os.getpid(),
                          started_at=now(), completed={})

    def save(self, name, value):
        path = self.output / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def publish(self):
        self.state["updated_at"] = now()
        self.save("queue_status.json", self.state)

    def event(self, event_name, **values):
        record = dict(event=event_name, time=now(), **values)
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)

    def request_stop(self, signum=None, _frame=None):
        self.stop_requested = True
        if self.stop_reason is None:
            self.stop_reason = f"signal {signum}" if signum is not None else "stop requested"

    def stopping(self):
        if (self.output / "STOP").exists():
            self.stop_requested = True
            self.stop_reason = self.stop_reason or "STOP file"
        return self.stop_requested

    def resources(self):
        memory = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                      if line.startswith("MemAvailable:"))
        return dict(disk_free_gib=shutil.disk_usage(self.output).free / 2**30,
                    memory_available_gib=memory / 2**30)

    def health(self, name):
        request = Request(f"http://127.0.0.1:{PORTS[name]}/health")
        token = os.environ.get("PRESSB_RL_TOKEN")
        if token:
            request.add_header("Authorization", "Bearer " + token)
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    def load_config(self):
        config = read(self.config_path)
        for key, expected in dict(method="action_residual", mode="train", seed=42,
                max_transitions=TRAINING_TRANSITIONS, eval_episodes=EVALUATION_EPISODES,
                learning_start=LEARNING_START, utd=1.0, checkpoint_every=100_000,
                rolling_checkpoint_every=10_000, pipeline_updates=True).items():
            if config.get(key) != expected:
                raise ValueError(f"Dedicated experiment config must set {key}={expected!r}")
        gamma = config.get("single_gamma")
        if type(gamma) not in (int, float) or not math.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError("single_gamma must be finite and in (0, 1]")
        learner = config.get("learner")
        scale = learner.get("residual_scale") if isinstance(learner, dict) else None
        residual_mode = learner.get("residual_mode", "pose9") if isinstance(learner, dict) else None
        if residual_mode not in ("pose9", "xyz"):
            raise ValueError("learner.residual_mode must be pose9 or xyz")
        residual_width = 3 if residual_mode == "xyz" else 9
        if (not isinstance(scale, list) or len(scale) != residual_width
                or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in scale)):
            raise ValueError(f"learner.residual_scale must contain {residual_width} finite positive numbers for {residual_mode}")
        baseline = read(DEFAULT_CONFIG)
        def fixed_fields(value):
            return {key: ({name: field for name, field in item.items()
                          if name not in ("residual_scale", "residual_mode")}
                          if key == "learner" else item)
                    for key, item in value.items() if key != "single_gamma"}
        if fixed_fields(config) != fixed_fields(baseline):
            raise ValueError("Only single_gamma, learner.residual_scale and learner.residual_mode may differ from the original 400k residual-on-noise config")
        return config

    def preflight(self):
        required = [PYTHON, INFERENCE_PYTHON, self.noise_checkpoint,
                    ROOT / "scripts/run_residual_on_noise.py", ROOT / "src/pressb/online_rl/residual_on_noise.py",
                    SERVICE_SOURCE / "scripts/serve_rl_fast_simulation.py",
                    SERVICE_SOURCE / "scripts/serve_rl_inference.py",
                    self.config_path, DEFAULT_CONFIG, OLD / "run_plan.json",
                    FROZEN_INPUT / "config.json", FROZEN_INPUT / "scene.usda",
                    FROZEN_INPUT / "assets/asset_bundle.json"]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Required experiment input does not exist: {path}")
        self.config = self.load_config()
        if self.stopping():
            raise InterruptedError(self.stop_reason)
        for name, port in PORTS.items():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
                connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    connection.bind(("127.0.0.1", port))
                except OSError as error:
                    raise RuntimeError(f"{name} port {port} is occupied; existing services will not be adopted") from error
        gpu_query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=20)
        gpu_rows = list(csv.reader(gpu_query.stdout.splitlines(), skipinitialspace=True))
        if not {"0", "1"}.issubset({row[0] for row in gpu_rows}):
            raise RuntimeError("Two local GPUs are required")
        processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=20)
        if processes.stdout.strip():
            raise RuntimeError("GPU compute processes are already running; this experiment will not stop or share them: "
                               + processes.stdout.strip())
        limits = self.resources()
        if limits["disk_free_gib"] < 32 or limits["memory_available_gib"] < 12:
            raise RuntimeError(f"Need at least 32 GiB free disk and 12 GiB available RAM at startup: {limits}")
        self.save("preflight.json", dict(checked_at=now(), resources=limits, gpu_rows=gpu_rows,
            compute_processes=[], ports_free=PORTS, minimum_free_disk_gib=32, minimum_available_memory_gib=12))

    def prepare(self):
        self.config = self.load_config()
        # Freeze the new training implementation while preserving the original
        # simulator/inference source hashes required by the noise checkpoint.
        shutil.copytree(ROOT / "src/pressb", self.source / "src/pressb",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        (self.source / "scripts").mkdir(parents=True)
        for name in ("run_residual_on_noise.py", "run_residual_on_noise_experiment.py",
                     "serve_rl_fast_simulation.py", "serve_rl_simulation.py", "serve_rl_inference.py",
                     "collect_dataset.py"):
            shutil.copy2(ROOT / "scripts" / name, self.source / "scripts" / name)
        sources = {str(path.relative_to(self.source)): file_identity(path)
                   for path in sorted(self.source.rglob("*")) if path.is_file()}
        self.save("source_manifest.json", sources)
        self.original_plan = read(OLD / "run_plan.json")
        self.noise_identity = file_identity(self.noise_checkpoint)
        shutil.copy2(self.config_path, self.output / "train_config.json")
        if read(self.output / "train_config.json") != self.config:
            raise ValueError("Selected config changed while preparing the frozen experiment")
        self.save("run_plan.json", dict(
            experiment="fresh_action_residual_on_frozen_trained_initial_noise",
            root=str(ROOT), run_root=str(self.output), source_root=str(self.source),
            service_source_root=str(SERVICE_SOURCE), noise_checkpoint=self.noise_identity,
            config_source=file_identity(self.config_path), single_gamma=self.config["single_gamma"],
            residual_scale=self.config["learner"]["residual_scale"],
            residual_mode=self.config["learner"].get("residual_mode", "pose9"),
            residual_action_dim=7 * len(self.config["learner"]["residual_scale"]),
            residual_initialization="fresh actor, critics, replay, optimizers, temperature, RNG and counters",
            warmstart_actor=None, resume=False,
            training_transitions=TRAINING_TRANSITIONS, learning_start=LEARNING_START,
            expected_updates=TRAINING_TRANSITIONS - LEARNING_START, utd=1.0,
            terminal_drain="Finish active episodes without inserting transitions or updating parameters after the 400000-transition budget",
            checkpoint_every=100_000, rolling_checkpoint_every=10_000,
            evaluation_checkpoint="train/last.pt", eval_episodes=EVALUATION_EPISODES,
            eval_seed=EVALUATION_SEED, train_seed=42,
            stages=["train", "eval"], num_envs=64, max_seconds=15,
            devices=dict(simulation_gpu=1, inference_gpu=1, learner_gpu=0), ports=PORTS,
            inference_batch_size=64, inference_cache_size=512, inference_cache_ttl=3600,
            camera_render_resolution=[320, 240], camera_image_resolution=[224, 224],
            resource_guard=dict(disk_free_gib=12, memory_available_gib=3),
            expected_new_disk_gib="approximately 15-20 including 4 archives, last.pt, atomic-save temporary file and logs",
            stop="Create STOP or send SIGTERM to supervisor: drain/checkpoint learner, skip eval, stop owned services",
            ownership="Each child PID, full argv and /proc starttime; no existing processes adopted",
            frozen_model_checkpoint_sha256=self.original_plan["checkpoint_sha256"],
            created_at=now()))

    def environment(self, gpu=None, inference=False):
        values = os.environ.copy()
        values.pop("PYTHONPATH", None)
        values.pop("CUDA_VISIBLE_DEVICES", None)
        values.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMNI_KIT_ACCEPT_EULA="YES")
        if gpu is not None:
            values["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if inference:
            values.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        return values

    def launch(self, name, argv, env):
        if self.stopping():
            raise InterruptedError(self.stop_reason)
        argv = list(map(str, argv))
        log = (self.output / f"{name}.log").open("ab", buffering=0)
        self.log_streams.append(log)
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        # Track the Popen immediately, including the brief exec()/proc race.
        child = dict(process=process, record=None)
        self.children[name] = child
        deadline = time.monotonic() + 3
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited during startup ({process.returncode}); see {name}.log")
            record = process_identity(process.pid)
            if record["argv"] == argv:
                child["record"] = record
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Could not establish identity of {name} PID {process.pid}")
            time.sleep(.01)
        self.save(f"{name}_process.json", record)
        self.event("process_started", name=name, **record)
        return child

    def start_services(self):
        plan = self.original_plan
        inference = [INFERENCE_PYTHON, "-u", SERVICE_SOURCE / "scripts/serve_rl_inference.py",
            "--vla-repo", plan["local_vla_repo"], "--checkpoint", plan["local_checkpoint"],
            "--device", "cuda:0", "--port", str(PORTS["inference"]), "--batch-size", "64",
            "--image-preprocess-device", "cuda", "--cache-size", "512", "--cache-ttl", "3600",
            "--torch-threads", "4", "--base-vlm", plan["base_vlm"], "--base-encoder", plan["base_encoder"]]
        simulation = [PYTHON, "-u", SERVICE_SOURCE / "scripts/serve_rl_fast_simulation.py",
            "--gpu", "1", "--port", str(PORTS["simulation"]), "--num-envs", "64", "--max-seconds", "15",
            "--camera-resolution", "224", "--ik-iterations", "16", "--output", self.output / "simulation",
            "--config", FROZEN_INPUT / "config.json", "--snapshot", FROZEN_INPUT / "scene.usda",
            "--dataset", FROZEN_INPUT / "dataset_metadata", "--asset-bundle", FROZEN_INPUT / "assets",
            "--single-gamma", str(self.config["single_gamma"]), "--gpu-dynamics"]
        self.launch("inference", inference, self.environment(1, inference=True))
        self.launch("simulation", simulation, self.environment())
        self.state.update(state="starting", stage="services")
        self.publish()
        deadline = time.monotonic() + 900
        ready = {}
        while len(ready) < 2:
            if self.stopping():
                raise InterruptedError(self.stop_reason)
            self.check_services()
            for name in ("simulation", "inference"):
                if name in ready:
                    continue
                try:
                    health = self.health(name)
                    if health.get("ready") is True:
                        ready[name] = health
                        self.save(f"node_{PORTS[name]}_health.json", health)
                except (OSError, ValueError):
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError("Services were not ready within 900 seconds")
            self.state["services_ready"] = list(ready)
            self.publish()
            if len(ready) < 2:
                time.sleep(3)
        sim, inference = ready["simulation"], ready["inference"]
        if (sim.get("num_envs") != 64 or sim.get("gpu") != 1 or sim.get("max_seconds") != 15
                or sim.get("run_id") is not None or sim.get("image_resolution") != [224, 224]
                or sim.get("render_resolution") != [320, 240] or sim.get("supports_indexed_reset") is not True
                or sim.get("gpu_dynamics") is not True
                or sim.get("single_gamma") != self.config["single_gamma"]
                or sim.get("physics_error_policy") != "fail_closed_on_native_physx_error"):
            raise ValueError("Simulation service does not match the frozen checkpoint's execution contract")
        original_sources = read(OLD / "source_manifest.json")
        if not sim.get("sources"):
            raise ValueError("Simulator did not report source identities")
        for name, identity in sim["sources"].items():
            if identity["sha256"] != original_sources[name]["sha256"]:
                raise ValueError(f"Simulator source differs from original checkpoint: {name}")
        if (inference.get("frozen") is not True or inference.get("checkpoint_sha256") != plan["checkpoint_sha256"]
                or inference.get("microbatch_size") != 64 or inference.get("image_preprocess_device") != "cuda"
                or inference.get("cache", {}).get("capacity") != 512
                or inference.get("cache", {}).get("ttl_seconds") != 3600
                or inference.get("cache", {}).get("size") != 0):
            raise ValueError("Inference service is not the required frozen, idle batched model")
        self.event("services_ready")

    def check_services(self):
        for name in ("simulation", "inference"):
            child = self.children.get(name)
            if child is None:
                continue
            if child["process"].poll() is not None or not owned(child["record"]):
                raise RuntimeError(f"Owned {name} service exited; stop learner and inspect {name}.log")
        # The frozen backend itself performs synchronous, incremental native
        # PhysX error checks and fails closed; do not repeatedly read its full log.

    def terminate(self, name, grace):
        child = self.children.get(name)
        if child is None:
            return
        process, record = child["process"], child["record"]
        if process.poll() is not None:
            return
        if record is not None and not owned(record):
            self.event("cleanup_identity_mismatch", name=name, pid=process.pid)
            return
        process.terminate()
        deadline = time.monotonic() + grace
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(.5)
        if process.poll() is None:
            if record is None or owned(record):
                process.kill()
                process.wait(timeout=30)
        self.event("process_stopped", name=name, pid=process.pid, returncode=process.poll())

    def run_stage(self, mode):
        if self.stopping():
            raise InterruptedError(self.stop_reason)
        if read(self.output / "train_config.json") != self.config:
            raise ValueError("Frozen train/eval configuration changed after preparation")
        argv = [PYTHON, "-u", self.source / "scripts/run_residual_on_noise.py",
            "--config", self.output / "train_config.json", "--noise-checkpoint", self.noise_checkpoint,
            "--simulation", f"http://127.0.0.1:{PORTS['simulation']}",
            "--inference", f"http://127.0.0.1:{PORTS['inference']}",
            "--output", self.output / mode, "--device", "cuda:0"]
        if mode == "eval":
            argv += ["--mode", "eval", "--checkpoint", self.output / "train/last.pt",
                     "--seed", str(EVALUATION_SEED), "--eval-episodes", str(EVALUATION_EPISODES)]
        self.learner = mode
        child = self.launch(mode, argv, self.environment(0))
        process = child["process"]
        self.state.update(state="running", stage=mode, learner_pid=process.pid)
        self.publish()
        stop_started = None
        while process.poll() is None:
            self.check_services()
            limits = self.resources()
            if limits["disk_free_gib"] < 12 or limits["memory_available_gib"] < 3:
                if not self.stop_requested:
                    self.event("resource_stop", **limits)
                self.stop_requested = True
                self.stop_reason = self.stop_reason or "resource guard"
            if self.stopping():
                if stop_started is None:
                    if owned(child["record"]):
                        process.terminate()
                    stop_started = time.monotonic()
                    self.state["state"] = "stopping"
                    self.event("graceful_stop_requested", stage=mode, reason=self.stop_reason)
                elif time.monotonic() - stop_started > 600:
                    raise TimeoutError("Learner did not drain/save within 600 seconds")
            status = self.output / mode / "status.json"
            if status.exists():
                self.state["learner"] = read(status)
            self.state["resources"] = limits
            self.publish()
            time.sleep(5)
        if process.returncode:
            raise RuntimeError(f"{mode} exited with code {process.returncode}; see {mode}.log")
        summary = read(self.output / mode / "summary.json")
        self.state["completed"][mode] = summary
        self.state["learner"] = summary
        self.publish()
        self.event("stage_finished", stage=mode, summary=summary)
        if self.stopping() or summary["state"] == "stopped":
            raise InterruptedError(self.stop_reason or "Learner stopped before completion")
        if summary["state"] != "complete":
            raise RuntimeError(f"Unexpected {mode} terminal state: {summary['state']}")
        manifest = read(self.output / mode / "manifest.json")
        if (manifest["config"].get("single_gamma") != self.config["single_gamma"]
                or manifest["identities"]["simulation"].get("single_gamma") != self.config["single_gamma"]
                or manifest["simulation"].get("single_gamma") != self.config["single_gamma"]):
            raise RuntimeError(f"{mode} did not use the selected discount consistently")
        if (manifest["config"].get("learner", {}).get("residual_scale") != self.config["learner"]["residual_scale"]
                or manifest["learner_config"].get("residual_scale") != self.config["learner"]["residual_scale"]):
            raise RuntimeError(f"{mode} did not use the selected residual scale consistently")
        selected_mode = self.config["learner"].get("residual_mode", "pose9")
        if (manifest["config"].get("learner", {}).get("residual_mode", "pose9") != selected_mode
                or manifest["learner_config"].get("residual_mode", "pose9") != selected_mode):
            raise RuntimeError(f"{mode} did not use the selected residual dimensions consistently")
        composition = read(self.output / mode / "composition.json")
        verification = read(self.output / mode / "freeze_verification.json")
        if (verification.get("noise_checkpoint_unchanged") is not True
                or verification.get("noise_parameters_updated") is not False
                or verification.get("base_parameters_updated") is not False
                or verification.get("noise_actor_sha256") != composition["frozen_noise"]["actor_sha256"]):
            raise RuntimeError("Run did not verify the unchanged frozen noise actor and VLA base")
        if mode == "train":
            if (composition.get("residual_initialization") != "from scratch"
                    or composition.get("residual_updates_at_start") != 0
                    or composition.get("replay_size_at_start") != 0):
                raise RuntimeError("Residual training did not start with fresh parameters and empty replay")
            if summary.get("training_transitions") != TRAINING_TRANSITIONS:
                raise RuntimeError("Training budget was not exactly 400000 inserted transitions")
            if summary.get("updates") != TRAINING_TRANSITIONS - LEARNING_START:
                raise RuntimeError("Expected exactly 398000 residual updates, excluding terminal drain")
            if not (self.output / "train/last.pt").is_file():
                raise RuntimeError("Final residual checkpoint is missing")
            events = [json.loads(line) for line in (self.output / "train/events.jsonl").read_text().splitlines() if line]
            budget_events = [event for event in events if event.get("event") == "training_budget_reached"]
            if len(budget_events) != 1:
                raise RuntimeError("Expected exactly one explicit training-budget boundary")
            budget = budget_events[0]
            if (budget["updates"] != TRAINING_TRANSITIONS - LEARNING_START
                    or budget["residual_actor_sha256"] != verification["residual_actor_sha256"]):
                raise RuntimeError("Residual parameters or update count changed during the final episode drain")
        elif summary["episodes"] != EVALUATION_EPISODES or summary["updates"] != 0:
            raise RuntimeError("Final evaluation must finish 120 episodes without parameter updates")
        if self.health("simulation").get("run_id") is not None:
            raise RuntimeError("Finished learner did not release simulation ownership")
        after = file_identity(self.noise_checkpoint)
        if after != self.noise_identity:
            raise RuntimeError("Frozen original noise checkpoint file changed")
        self.save(f"{mode}_noise_checkpoint_verification.json", dict(unchanged=True, before=self.noise_identity, after=after))
        return summary

    def compare_evaluation(self):
        inputs = {"base": OLD / "base_eval", "original_residual": OLD / "action_residual_eval",
                  "frozen_noise": OLD / "initial_noise_eval", "residual_on_frozen_noise": self.output / "eval"}
        combined = ROOT / "outputs/online_rl_combined_eval_20261006/combined_eval"
        if (combined / "summary.json").is_file():
            inputs["independently_trained_combination"] = combined
        previous = ROOT / "outputs/online_rl_residual_on_noise_400k_20261006/eval"
        if previous.resolve() != (self.output / "eval").resolve() and (previous / "summary.json").is_file():
            inputs["residual_on_frozen_noise_gamma099"] = previous
        previous_fullscale = ROOT / "outputs/online_rl_residual_on_noise_gamma0995_400k_20261007/eval"
        if (previous_fullscale.resolve() != (self.output / "eval").resolve()
                and (previous_fullscale / "summary.json").is_file()):
            inputs["residual_on_frozen_noise_gamma0995_fullscale"] = previous_fullscale
        previous_halfscale = ROOT / "outputs/online_rl_residual_on_noise_gamma0995_scale050_400k_20261008/eval"
        if (previous_halfscale.resolve() != (self.output / "eval").resolve()
                and (previous_halfscale / "summary.json").is_file()):
            inputs["residual_on_frozen_noise_gamma0995_halfscale"] = previous_halfscale
        positions = {(0., 0.), (-.01, -.025), (-.01, .025), (.01, -.025), (.01, .025)}
        result = {}
        reference_identity = read(self.output / "eval/manifest.json")["identities"]
        for method, directory in inputs.items():
            summary = read(directory / "summary.json")
            manifest = read(directory / "manifest.json")
            episodes = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines() if line]
            if len(episodes) != 120 or len({row["episode_id"] for row in episodes}) != 120:
                raise ValueError(f"Incomplete evaluation: {method}")
            if summary["episodes"] != 120 or summary["updates"] != 0 or manifest["config"]["seed"] != EVALUATION_SEED:
                raise ValueError(f"Evaluation protocol differs: {method}")
            # Binary task success is independent of return discounting. Permit
            # only this explicit gamma exception; preserve every other model
            # and physical execution identity, including source hashes.
            identities = manifest["identities"]
            if identities["inference"] != reference_identity["inference"]:
                raise ValueError(f"Evaluation inference identity differs: {method}")
            simulation = identities["simulation"]
            gamma = simulation.get("single_gamma")
            if (type(gamma) not in (int, float) or not math.isfinite(gamma) or not 0 < gamma <= 1
                    or manifest["config"].get("single_gamma") != gamma
                    or manifest["simulation"].get("single_gamma") != gamma):
                raise ValueError(f"Evaluation discount identity is invalid or inconsistent: {method}")
            without_gamma = lambda value: {key: item for key, item in value.items() if key != "single_gamma"}
            if without_gamma(simulation) != without_gamma(reference_identity["simulation"]):
                raise ValueError(f"Evaluation simulation identity differs beyond single_gamma: {method}")
            per_floor = {}
            for floor in range(24, 36):
                selected = [row for row in episodes if row["layout"]["floor"] == floor]
                counts = {(x, y): sum(row["layout"]["offset_x_m"] == x and row["layout"]["offset_y_m"] == y
                                     for row in selected) for x, y in positions}
                if len(selected) != 10 or set(counts.values()) != {2}:
                    raise ValueError(f"Missing fixed floor/layout/repeat coverage: {method}/{floor}")
                successes = sum(bool(row["success"]) for row in selected)
                per_floor[str(floor)] = dict(successes=successes, episodes=10, success_rate=successes / 10)
            successes = sum(bool(row["success"]) for row in episodes)
            if successes != summary["successes"]:
                raise ValueError(f"Success count disagrees with summary: {method}")
            residual_scale = None
            residual_mode = None
            if manifest["config"]["method"] == "action_residual":
                residual_scale = manifest["learner_config"]["residual_scale"]
                residual_mode = manifest["learner_config"].get("residual_mode", "pose9")
            elif method == "independently_trained_combination":
                residual_config = read(directory / "composition.json")["residual_config"]
                residual_scale = residual_config["residual_scale"]
                residual_mode = residual_config.get("residual_mode", "pose9")
            result[method] = dict(directory=str(directory), single_gamma=gamma,
                                 residual_scale=residual_scale, residual_mode=residual_mode,
                                 successes=successes, episodes=120,
                                 success_rate=successes / 120, per_floor=per_floor)
        self.save("evaluation_comparison.json", dict(methods=result,
            metric="Undiscounted binary task success; discounted returns are not compared",
            identity_exception="Only simulation.single_gamma may differ; all other simulation and inference identity fields must match",
            caveat="Fixed 120 prescribed conditions and seed; separate executions are not guaranteed bitwise identical.",
            checkpoint_selection="Final last.pt after exactly 400000 residual training transitions; no selection on these evaluation results."))
        with (self.output / "per_button_comparison.csv").open("w", newline="") as stream:
            columns = ["floor"] + [f"{method}_{field}" for method in inputs for field in ("single_gamma", "successes", "episodes", "success_rate")]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for floor in range(24, 36):
                writer.writerow(dict(floor=floor,
                    **{f"{method}_single_gamma": result[method]["single_gamma"] for method in inputs},
                    **{f"{method}_{field}": result[method]["per_floor"][str(floor)][field]
                       for method in inputs for field in ("successes", "episodes", "success_rate")}))

    def cleanup(self):
        # Give a responsive learner time to drain/save. If a service died, its
        # RPC cannot complete reliably; terminate the learner with a short grace.
        services_alive = all(name in self.children and self.children[name]["process"].poll() is None
                             for name in ("simulation", "inference"))
        errors = []
        targets = ([(self.learner, 600 if services_alive else 20)] if self.learner else [])
        targets += [("simulation", 120), ("inference", 60)]
        for name, grace in targets:
            try:
                self.terminate(name, grace)
            except BaseException as error:
                errors.append(f"{name}: {type(error).__name__}: {error}")
        for stream in self.log_streams:
            stream.close()
        if errors:
            raise RuntimeError("; ".join(errors))

    def run(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.request_stop)
        exit_code = 0
        try:
            self.publish()
            self.preflight()
            self.prepare()
            self.start_services()
            self.run_stage("train")
            self.run_stage("eval")
            self.compare_evaluation()
            self.state.update(state="complete", stage="complete")
        except InterruptedError as error:
            self.state.update(state="stopped", reason=str(error))
            self.event("experiment_stopped", reason=str(error))
        except BaseException as error:
            exit_code = 1
            self.state.update(state="failed", error=f"{type(error).__name__}: {error}")
            self.event("experiment_failed", error=self.state["error"])
            traceback.print_exc()
        finally:
            try:
                self.cleanup()
            except BaseException as error:
                exit_code = 1
                self.state.update(state="failed", cleanup_error=f"{type(error).__name__}: {error}")
                traceback.print_exc()
            self.state["finished_at"] = now()
            self.publish()
            self.event("supervisor_exit", state=self.state["state"])
        return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New or empty experiment directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="400k experiment config; only single_gamma, learner.residual_scale and learner.residual_mode may differ from the default")
    parser.add_argument("--noise-checkpoint", type=Path, default=OLD / "initial_noise_train/last.pt")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "supervisor.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another supervisor owns this output directory")
        if any(path.name != "supervisor.lock" for path in output.iterdir()):
            parser.error("Output directory must be new or empty; existing experiment results will not be overwritten")
        return Experiment(output, args.noise_checkpoint, args.config).run()


if __name__ == "__main__":
    raise SystemExit(main())
