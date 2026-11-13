"""Supervise independent evaluation processes sharing one frozen model service.

Task/model factories, physics, policy inference, and episode bookkeeping remain
in the existing experiment nodes and GenericTrainer. Workers only synchronize
at service startup and final aggregation, never between action chunks. No reset,
step, failed worker, or interrupted run is retried by this supervisor.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

from pressb.experiments.config import PROJECT_ROOT, load_experiment


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def sha256_file(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def split_episodes(episodes, workers):
    if (type(episodes) is not int or type(workers) is not int
            or not 1 <= workers <= episodes):
        raise ValueError("episodes and workers must be positive integers, with workers <= episodes")
    quotient, remainder = divmod(episodes, workers)
    return tuple(quotient + (index < remainder) for index in range(workers))


def validate_args(args):
    counts = split_episodes(args.episodes, args.workers)
    for key in ("num_envs", "batch_size"):
        if type(getattr(args, key)) is not int or getattr(args, key) < 1:
            raise ValueError(f"{key} must be a positive integer")
    if min(counts) < args.num_envs:
        raise ValueError("Each worker needs at least num_envs allocated episodes; lower --num-envs or increase --episodes")
    for key in ("sim_gpu", "inference_gpu"):
        if type(getattr(args, key)) is not int or getattr(args, key) < 0:
            raise ValueError(f"{key} must be a nonnegative physical GPU index")
    # Same GPU is allowed for explicit diagnostics; one inference process still
    # owns the sole model instance. GPU availability is the caller's decision.
    ports = [args.simulation_port + i for i in range(args.workers)] + [args.inference_port]
    if any(type(port) is not int or not 1024 <= port <= 65535 for port in ports) or len(set(ports)) != len(ports):
        raise ValueError("Simulation worker ports and inference port must be distinct unprivileged ports")
    if args.seed is not None and (type(args.seed) is not int or not 0 <= args.seed < 2**32):
        raise ValueError("seed must be an integer in [0, 2**32)")
    for key in ("startup_timeout", "poll_seconds", "shutdown_timeout"):
        value = getattr(args, key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if args.max_transitions_per_worker is not None and (
            type(args.max_transitions_per_worker) is not int or args.max_transitions_per_worker < 1):
        raise ValueError("max_transitions_per_worker must be a positive integer")
    if args.expected_weights_sha256 is not None and not re.fullmatch("[0-9a-f]{64}", args.expected_weights_sha256):
        raise ValueError("expected-weights-sha256 must be a lowercase SHA256 digest")
    if args.expected_task_id is not None and not args.expected_task_id.strip():
        raise ValueError("expected-task-id must be nonempty")
    if args.startup_order not in ("inference-first", "parallel"):
        raise ValueError("startup_order must be inference-first or parallel")


def _checkpoint_expectation(model_config, explicit):
    """Use conventional local weight files when present, without model imports.

Unconventional/remote factories retain their own checkpoint validation and may
be pinned explicitly with --expected-weights-sha256. Their health identity is
recorded, but is not represented as an independently verified local digest.
"""
    checkpoint = model_config.get("checkpoint")
    if checkpoint is None and model_config.get("run"):
        candidate = Path(model_config["run"]).expanduser() / "checkpoints/best"
        if candidate.exists():
            checkpoint = str(candidate)
    candidates = []
    if isinstance(checkpoint, str):
        path = Path(checkpoint).expanduser()
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = [path / name for name in ("model.safetensors", "model.pt", "pytorch_model.bin")
                          if (path / name).is_file()]
    if len(candidates) == 1:
        digest = sha256_file(candidates[0])
        if explicit is not None and explicit != digest:
            raise ValueError("Selected local checkpoint differs from expected weights SHA256")
        return {"weights_sha256": digest, "weight_file": str(candidates[0].resolve()),
                "verification": "local_weights_sha256"}
    return {"weights_sha256": explicit,
            "verification": "explicit_weights_sha256" if explicit else "adapter_reported_identity"}


def build_plan(args):
    """Resolve inheritance once and return JSON-serializable worker documents."""
    validate_args(args)
    source = load_experiment(args.config)
    original = source["experiment"]
    run = source["nodes"]["trainer"]
    if run.get("method") != "base" or run.get("mode") != "eval":
        raise ValueError("Sharded evaluation currently requires run.method=base and run.mode=eval")
    factory = source["factories"]["task"]
    task = original["task"]
    task_config = task.get("config", {})
    if (task.get("plugin") == "piper_real" or "real_robot" in factory
            or task_config.get("environment_type") == "real_robot"
            or task_config.get("backend") == "physical_robot"):
        raise ValueError("Sharded evaluation does not launch physical robot backends")
    seed = run.get("seed", 42) if args.seed is None else args.seed
    if type(seed) is not int or not 0 <= seed < 2**32 or seed + args.workers > 2**32:
        raise ValueError("Worker seeds must fit in [0, 2**32) without wrapping")
    output = Path(args.output).expanduser().resolve()
    expected_task = args.expected_task_id or task_config.get("task_id")
    if expected_task is None and task.get("plugin") not in (None, "pap_process", "pap_recording", "elevator_remote"):
        expected_task = task["plugin"]
    checkpoint = _checkpoint_expectation(original["model"].get("config", {}), args.expected_weights_sha256)
    from pressb.runtime.trainer import TrainerConfig

    workers, offset = [], 0
    for index, count in enumerate(split_episodes(args.episodes, args.workers)):
        worker_dir = output / "workers" / f"{index:03d}"
        data = deepcopy(original)
        data["name"] = f"{source['name']}_worker_{index:03d}"
        data["task"].setdefault("config", {}).update(
            num_envs=args.num_envs, gpu=args.sim_gpu, log_dir=str(worker_dir / "simulation"))
        data["model"].setdefault("config", {})["device"] = f"cuda:{args.inference_gpu}"
        nodes = data.setdefault("nodes", {})
        for kind, port in (("simulation", args.simulation_port + index), ("inference", args.inference_port)):
            node = nodes.setdefault(kind, {})
            node.setdefault("server", {}).update(host="127.0.0.1", port=port)
        inference = nodes["inference"].setdefault("service", {})
        inference.update(batch_size=args.batch_size, collect_timing=True,
                         cache_size=max(inference.get("cache_size", 0), 2 * args.workers * args.num_envs))
        worker_run = deepcopy(run)
        worker_run.update(output_dir=str(worker_dir / "run"), seed=seed + index)
        endpoints = worker_run.setdefault("endpoints", {})
        for kind in ("simulation", "inference"):
            server = nodes[kind]["server"]
            transport = server.get("transport", "ws" if kind == "inference" else "http")
            endpoints[kind] = f"{transport}://127.0.0.1:{server['port']}"
        rollout = worker_run.setdefault("rollout", {})
        rollout.update(eval_episodes=count, num_envs=args.num_envs)
        if args.max_transitions_per_worker is not None:
            rollout["max_transitions"] = args.max_transitions_per_worker
        data["run"] = worker_run
        TrainerConfig(**deepcopy(worker_run))
        workers.append(dict(worker_id=index, episodes=count, global_episode_start=offset,
                            seed=seed + index, config=str(worker_dir / "experiment.json"),
                            run_dir=str(worker_dir / "run"), experiment=data))
        offset += count
    return dict(schema_version=1, source_config=str(Path(args.config).resolve()), sources=source["sources"],
                task_factory=factory, model_factory=source["factories"]["model"], expected_task_id=expected_task,
                checkpoint=checkpoint, requested_episodes=args.episodes, num_envs_per_worker=args.num_envs,
                batch_size=args.batch_size, sim_gpu=args.sim_gpu, inference_gpu=args.inference_gpu,
                output=str(output), workers=workers, diagnostic=args.max_transitions_per_worker is not None,
                startup_order=args.startup_order,
                seed_semantics="Independent seed + worker_id RNG streams. Changing worker count or full/fused terminal scheduling changes layout-reset and policy-noise streams; equal initial seeds do not imply paired later episodes. Global IDs are assigned by each worker's completion order.")


def _sum_numeric(target, values):
    for key, value in values.items():
        if type(value) in (int, float) and math.isfinite(value):
            target[key] = target.get(key, 0) + value


def summarize(plan, *, wall_seconds, startup_seconds=0., diagnostic=None):
    """Return a summary and exact completed rows, counting failures in the denominator."""
    diagnostic = plan["diagnostic"] if diagnostic is None else diagnostic
    all_rows, workers, rpc, counters, status_seconds = [], [], {}, {}, {}
    total_worker_seconds = 0.
    reasons = Counter()
    for worker in plan["workers"]:
        run_dir = Path(worker["run_dir"])
        status = read_json(run_dir / "status.json", {})
        rows = read_rows(run_dir / "episodes.jsonl")
        allowed = ("completed", "transition_limit") if diagnostic else ("completed",)
        if status.get("state") not in allowed:
            raise RuntimeError(f"Worker {worker['worker_id']} did not complete: {status.get('state')}")
        expected = worker["episodes"]
        if (len(rows) != status.get("episodes") or len(rows) > expected
                or (not diagnostic and len(rows) != expected)
                or (status.get("state") == "completed" and len(rows) != expected)):
            raise RuntimeError(f"Worker {worker['worker_id']} episode count differs from its allocation ({len(rows)}/{expected})")
        errors = [row for row in read_rows(run_dir / "events.jsonl")
                  if row.get("event") in ("context_cleanup_failed", "simulation_close_failed", "observer_close_failed")]
        if status.get("context_cleanup_error") or errors:
            raise RuntimeError(f"Worker {worker['worker_id']} cleanup failed: {errors or status['context_cleanup_error']}")
        identities = set()
        for index, row in enumerate(rows):
            identity = (row["env_id"], row["episode_id"])
            if identity in identities:
                raise ValueError(f"Repeated completed episode identity in worker {worker['worker_id']}")
            identities.add(identity)
            if type(row.get("success")) is not bool:
                raise ValueError("Every completed episode must explicitly record boolean success")
            physical = row.get("info", {}).get("physical_outcome", {})
            if "success" in physical and physical["success"] is not row["success"]:
                raise ValueError("Episode success differs from physical outcome")
            all_rows.append(dict(row, worker_id=worker["worker_id"], local_episode_id=row["episode_id"],
                                 global_episode_id=worker["global_episode_start"] + index))
            info = row.get("info", {})
            reasons[str(info.get("terminal_reason", info.get("termination_reason", "unspecified")))] += 1
        successes = sum(row["success"] for row in rows)
        if successes != status.get("successes"):
            raise ValueError(f"Worker {worker['worker_id']} success summary disagrees with episode rows")
        total_worker_seconds += status.get("elapsed_seconds", 0.)
        _sum_numeric(status_seconds, status.get("rpc_seconds", {}))
        for row in read_rows(run_dir / "rpc_timing.jsonl"):
            name = f"{row['node']}{row['path']}"
            group = rpc.setdefault(name, dict(calls=0, elapsed_seconds=0., client={}, server={}))
            group["calls"] += 1
            group["elapsed_seconds"] += row["elapsed_seconds"]
            for kind in ("client", "server"):
                _sum_numeric(group[kind], row.get(kind, {}))
        snapshot = read_json(run_dir.parent / "simulation.final_status.json", {})
        if snapshot.get("fault"):
            raise RuntimeError(f"Worker {worker['worker_id']} simulator fault: {snapshot['fault']}")
        backend = snapshot.get("backend", {})
        timing = backend.get("timings", backend.get("timing", {}))
        _sum_numeric(counters, timing)
        workers.append(dict(worker_id=worker["worker_id"], seed=worker["seed"], requested_episodes=expected,
                            episodes=len(rows), successes=successes, runner_status=status,
                            simulation_status=snapshot))
    completed = len(all_rows)
    if not diagnostic and {row["global_episode_id"] for row in all_rows} != set(range(plan["requested_episodes"])):
        raise ValueError("Global episode IDs must exactly cover the requested evaluation")
    successes = sum(row["success"] for row in all_rows)
    rate = successes / completed if completed else None
    interval = None
    if completed:
        z = 1.959963984540054
        center = (rate + z*z/(2*completed)) / (1+z*z/completed)
        radius = z*math.sqrt(rate*(1-rate)/completed+z*z/(4*completed**2)) / (1+z*z/completed)
        interval = [center-radius, center+radius]
    summary = dict(schema_version=1, status="diagnostic_completed" if diagnostic else "completed",
                   formal_evaluation=not diagnostic, requested_episodes=plan["requested_episodes"],
                   episodes=completed, successes=successes, failures=completed-successes,
                   success_rate=rate, success_rate_wilson95=interval, termination_reasons=dict(reasons),
                   success_rate_scope="Completed episodes only; transition-limited probes are not formal evaluation estimates" if diagnostic else "All requested episodes, including failures",
                   workers=workers, inference_processes=1, checkpoint=plan["checkpoint"],
                   startup_order=plan.get("startup_order", "inference-first"),
                   timing=dict(evaluation_wall_seconds=wall_seconds, startup_wall_seconds=startup_seconds,
                               worker_elapsed_seconds_sum=total_worker_seconds,
                               rpc_seconds_worker_sum=status_seconds, rpc_successful_calls=rpc,
                               simulation_counters_worker_sum=counters,
                               semantics="Wall time is the concurrent critical path; worker/RPC sums overlap. Simulation counters can nest (e.g. contact callbacks within physics) and must not be summed into wall time."),
                   seed_semantics=plan["seed_semantics"])
    return summary, all_rows


class ShardedEvaluation:
    def __init__(self, args):
        self.args = args
        self.output = Path(args.output).expanduser().resolve()
        self.children = {}
        self.logs = []
        self.verified_workers = set()
        self.stopping = False
        self.plan = None
        self.state = dict(status="preparing", pid=os.getpid(), started_unix=time.time(), processes={})

    def update(self, state, **extra):
        self.state.update(status=state, updated_unix=time.time(), **extra)
        write_json(self.output / "status.json", self.state)
        print(json.dumps(dict(status=state, **extra), allow_nan=False), flush=True)

    def check_children(self):
        if self.stopping:
            raise InterruptedError("Sharded evaluation received SIGINT/SIGTERM")
        for name, child in self.children.items():
            code = child.poll()
            if code is not None and not self.expected_exit(name, code):
                raise RuntimeError(f"Owned {name} process exited with code {code}; see {self.state['processes'][name]['log']}")
            if code is not None and name.startswith("eval_") and name not in self.verified_workers:
                worker = self.plan["workers"][int(name.removeprefix("eval_"))]
                directory = Path(worker["run_dir"])
                status = read_json(directory / "status.json", {})
                allowed = ("completed", "transition_limit") if self.plan["diagnostic"] else ("completed",)
                if status.get("state") not in allowed:
                    raise RuntimeError(f"Worker {worker['worker_id']} stopped without completion: {status.get('state')}")
                errors = [row for row in read_rows(directory / "events.jsonl")
                          if row.get("event") in ("context_cleanup_failed", "simulation_close_failed", "observer_close_failed")]
                if errors or status.get("context_cleanup_error"):
                    raise RuntimeError(f"Worker {worker['worker_id']} cleanup failed: {errors or status['context_cleanup_error']}")
                self.verified_workers.add(name)

    def expected_exit(self, name, code):
        # The CLI's explicit diagnostic flag accepts transition_limit only
        # after GenericTrainer.run() returns normally. Never reinterpret a
        # nonzero exception/cleanup exit as a successful throughput probe.
        return name.startswith("eval_") and code == 0

    def pause(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.check_children()
            time.sleep(min(.2, max(0., deadline-time.monotonic())))

    def launch(self, name, node, python, worker):
        self.check_children()
        directory = Path(worker["config"]).parent if node != "inference" else self.output
        command = [str(python), "-u", str(PROJECT_ROOT / "scripts/run_experiment.py"), node,
                   "--config", worker["config"], "--resolved-dir", str(directory / "resolved" / node)]
        if node == "eval" and self.plan["diagnostic"]:
            command.append("--allow-transition-limit")
        log_path = directory / f"{node}.log"
        log = log_path.open("x")
        self.logs.append(log)
        env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
        # GPU indices are physical local indices, consistent with the CLI.
        env.pop("CUDA_VISIBLE_DEVICES", None)
        child = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.children[name] = child
        self.state["processes"][name] = dict(pid=child.pid, command=command, log=str(log_path))
        self.update("evaluating" if node == "eval" else "starting_nodes")

    def call(self, endpoint, path, timeout=3.):
        from pressb.runtime.client import NodeClient
        client = NodeClient(endpoint, timeout=timeout, token=os.environ.get("PRESSB_RL_TOKEN"))
        try:
            return client.call(path)
        finally:
            client.close()

    def validate_health(self, health, kind, worker):
        from pressb.runtime.contract import validate_metadata
        validate_metadata(health, kind)
        if health.get("ready") is False:
            raise ValueError("Node is not ready")
        if kind == "inference":
            expected = self.plan["checkpoint"]["weights_sha256"]
            if expected and expected not in (health.get("weights_sha256"), health.get("checkpoint_sha256")):
                raise ValueError("Inference checkpoint hash differs from selected weights")
            configured_id = worker["experiment"]["model"].get("config", {}).get("model_id")
            if configured_id and health.get("model_id") != configured_id:
                raise ValueError("Inference model_id differs from configured model")
            if health.get("frozen") is False:
                raise ValueError("Base evaluation requires frozen inference")
            if health.get("cache", {}).get("capacity", 0) < 2 * len(self.plan["workers"]) * self.args.num_envs:
                raise ValueError("Inference cache cannot hold two observation batches per worker")
            if health.get("microbatch_size") != self.args.batch_size:
                raise ValueError("Inference microbatch size differs from the requested size")
            return
        if health.get("environment_type") == "real_robot" or health.get("execution", {}).get("backend") == "physical_robot":
            raise ValueError("Sharded evaluation refuses physical robot backends")
        if self.plan["expected_task_id"] and health.get("task_id") != self.plan["expected_task_id"]:
            raise ValueError("Simulation task_id differs from selected task")
        if health.get("num_envs") != self.args.num_envs:
            raise ValueError("Simulator environment count differs from worker configuration")
        config = worker["experiment"]["task"].get("config", {})
        for key, reported in (("horizon", health["action"]["horizon"]),
                              ("control_hz", health["action"]["control_hz"]),
                              ("max_episode_seconds", health.get("execution", {}).get("max_episode_seconds"))):
            if key in config and reported is not None and config[key] != reported:
                raise ValueError(f"Simulation {key} differs from selected task configuration")
        gpu_ids = health.get("execution", {}).get("gpu_ids")
        if gpu_ids is not None and gpu_ids != [self.args.sim_gpu]:
            raise ValueError("Simulator reports a different GPU")
        source_root = PROJECT_ROOT / "src/pressb"
        for relative, expected in health.get("source_identity", {}).items():
            path = (source_root / relative).resolve()
            if not path.is_relative_to(source_root) or not path.is_file() or sha256_file(path) != expected.get("sha256"):
                raise ValueError(f"Simulation runtime source differs from workspace: {relative}")

    def await_health(self, *, inference_only=False, deadline=None):
        from pressb.runtime.contract import negotiate
        workers = self.plan["workers"]
        pending = {"inference": ("inference", workers[0])}
        if not inference_only:
            pending.update({f"simulation_{w['worker_id']:03d}": ("simulation", w) for w in workers})
        results, errors = {}, {}
        if deadline is None:
            deadline = time.monotonic() + self.args.startup_timeout
        while pending:
            self.check_children()
            for name, (kind, worker) in list(pending.items()):
                endpoint = worker["experiment"]["run"]["endpoints"][kind]
                try:
                    health = self.call(endpoint, "/health", timeout=min(3., max(.1, deadline-time.monotonic())))
                except Exception as error:
                    errors[name] = str(error)
                    continue
                # A responsive wrong node is an immediate failure, not readiness
                # polling. Only read-only unavailable-health calls are repeated.
                self.validate_health(health, kind, worker)
                directory = self.output if kind == "inference" else Path(worker["config"]).parent
                write_json(directory / f"{kind}.health.json", health)
                results[name] = health
                del pending[name]
            if pending and time.monotonic() >= deadline:
                raise TimeoutError(f"Services not ready: {list(pending)}; last errors={errors}")
            if pending:
                self.pause(min(1., max(.01, deadline-time.monotonic())))
        # A signal or process exit may happen during the last successful health
        # call. Do not advance to the next startup stage after either event.
        self.check_children()
        if inference_only:
            return results["inference"]
        reference = results["simulation_000"]
        for worker in workers:
            simulation = results[f"simulation_{worker['worker_id']:03d}"]
            for key in ("task_id", "task_config_sha256", "source_revision", "reward_schema", "success_rule", "outcome_thresholds"):
                if simulation.get(key) != reference.get(key):
                    raise ValueError(f"Simulator workers disagree on {key}")
            rollout = worker["experiment"]["run"].get("rollout", {})
            negotiate(results["inference"], simulation, "base", rollout.get("execution_horizon"),
                      terminal_observation_mode=rollout.get("terminal_observation_mode", "full"),
                      fused_step_reset=rollout.get("fused_step_reset", False))
        self.state["health_verified"] = True

    def progress(self):
        workers = [dict(worker_id=worker["worker_id"], requested_episodes=worker["episodes"],
                        **read_json(Path(worker["run_dir"]) / "status.json", {})) for worker in self.plan["workers"]]
        episodes = sum(row.get("episodes", 0) for row in workers)
        successes = sum(row.get("successes", 0) for row in workers)
        return dict(workers=workers, episodes=episodes, successes=successes, failures=episodes-successes,
                    requested_episodes=self.plan["requested_episodes"], success_rate=successes/episodes if episodes else None)

    def final_snapshots(self):
        for worker in self.plan["workers"]:
            endpoints = worker["experiment"]["run"]["endpoints"]
            status = self.call(endpoints["simulation"], "/status", timeout=10.)
            if status.get("fault"):
                raise RuntimeError(f"Simulator {worker['worker_id']} is faulted: {status['fault']}")
            if status.get("run_id") is not None:
                raise RuntimeError(f"Worker {worker['worker_id']} failed to close its simulator session")
            directory = Path(worker["config"]).parent
            write_json(directory / "simulation.final_status.json", status)
            write_json(directory / "simulation.final_health.json", self.call(endpoints["simulation"], "/health", timeout=10.))
        inference = self.call(self.plan["workers"][0]["experiment"]["run"]["endpoints"]["inference"], "/health", timeout=10.)
        if inference.get("cache", {}).get("size") != 0:
            raise RuntimeError("Workers left live inference contexts after completion")
        write_json(self.output / "inference.final_health.json", inference)

    def cleanup(self):
        errors = []
        for name, child in self.children.items():
            if child.poll() is not None:
                continue
            try:
                # Only unreaped live children can establish current ownership of
                # their process groups; never signal a possibly reused old PID.
                os.killpg(child.pid, signal.SIGTERM if name.startswith("eval_") else signal.SIGINT)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + self.args.shutdown_timeout
        for name, child in self.children.items():
            try:
                code = child.wait(timeout=max(.1, deadline-time.monotonic()))
                if self.state.get("status") == "finalizing" and code != 0 and not self.expected_exit(name, code):
                    errors.append(f"{name} shutdown exited with code {code}")
            except subprocess.TimeoutExpired:
                errors.append(f"{name} did not stop within the shutdown timeout")
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=10)
        for log in self.logs:
            log.close()
        self.state["children_cleaned_up"] = True
        if errors:
            self.state["cleanup_errors"] = errors
        return errors

    def run(self):
        started = time.monotonic()
        cleaned = False
        try:
            self.update("preparing")
            self.plan = build_plan(self.args)
            for worker in self.plan["workers"]:
                write_json(worker["config"], worker["experiment"])
                # Recompile generated configs using the exact node launch path.
                load_experiment(worker["config"])
            write_json(self.output / "plan.json", self.plan)
            if self.args.validate_only:
                self.update("validated", gpu_processes_started=False, plan=str(self.output / "plan.json"))
                return 0
            for executable in (self.args.sim_python, self.args.model_python):
                if not Path(executable).is_file() or not os.access(executable, os.X_OK):
                    raise FileNotFoundError(f"Python interpreter is not executable: {executable}")
            # Check every address before starting any node; an occupied endpoint
            # must never be silently reused as an externally owned service.
            sockets = []
            try:
                for port in [self.args.simulation_port + i for i in range(self.args.workers)] + [self.args.inference_port]:
                    sock = socket.socket()
                    sockets.append(sock)
                    sock.bind(("127.0.0.1", port))
            finally:
                for sock in sockets:
                    sock.close()
            node_startup = time.monotonic()
            deadline = node_startup + self.args.startup_timeout
            self.launch("inference", "inference", self.args.model_python, self.plan["workers"][0])
            if self.args.startup_order == "inference-first":
                self.update("waiting_inference", startup_order=self.args.startup_order)
                self.await_health(inference_only=True, deadline=deadline)
                self.update("inference_ready", inference_startup_wall_seconds=time.monotonic()-node_startup)
            for worker in self.plan["workers"]:
                self.launch(f"simulation_{worker['worker_id']:03d}", "simulation", self.args.sim_python, worker)
            self.await_health(deadline=deadline)
            startup_seconds = time.monotonic()-started
            evaluation_started = time.monotonic()
            for worker in self.plan["workers"]:
                self.launch(f"eval_{worker['worker_id']:03d}", "eval", self.args.model_python, worker)
            next_report = 0.
            while any(child.poll() is None for name, child in self.children.items() if name.startswith("eval_")):
                self.check_children()
                if time.monotonic() >= next_report:
                    self.update("evaluating", progress=self.progress(), evaluation_wall_seconds=time.monotonic()-evaluation_started)
                    next_report = time.monotonic()+self.args.poll_seconds
                self.pause(.2)
            self.check_children()
            wall_seconds = time.monotonic()-evaluation_started
            self.final_snapshots()
            result, rows = summarize(self.plan, wall_seconds=wall_seconds, startup_seconds=startup_seconds)
            self.update("finalizing", progress=self.progress())
            cleanup_errors = self.cleanup()
            cleaned = True
            if cleanup_errors:
                raise RuntimeError("Owned child cleanup failed: " + "; ".join(cleanup_errors))
            result["timing"]["total_wall_seconds"] = time.monotonic()-started
            if "inference_startup_wall_seconds" in self.state:
                result["timing"]["inference_startup_wall_seconds"] = self.state["inference_startup_wall_seconds"]
            result["finished_unix"] = time.time()
            with (self.output / "episodes.jsonl").open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
            write_json(self.output / "result.json", result)
            self.update(result["status"], episodes=result["episodes"], successes=result["successes"],
                        failures=result["failures"], success_rate=result["success_rate"],
                        formal_evaluation=result["formal_evaluation"], result=str(self.output / "result.json"))
            return 0
        except BaseException as error:
            self.update("stopped" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                        error_type=type(error).__name__, error=str(error))
            raise
        finally:
            if not cleaned:
                self.cleanup()
            write_json(self.output / "status.json", self.state)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", "--simulation-workers", dest="workers", type=int, default=2)
    parser.add_argument("--num-envs", type=int, default=8, help="Environment slots per independent simulator")
    parser.add_argument("--episodes", type=int, default=100, help="Total completed episodes, including failures")
    parser.add_argument("--sim-gpu", type=int, default=0)
    parser.add_argument("--inference-gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--simulation-port", type=int, default=18720, help="First port; workers use consecutive ports")
    parser.add_argument("--inference-port", type=int, default=18750)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory; interrupted runs are never resumed")
    parser.add_argument("--sim-python", type=Path, default=PROJECT_ROOT / ".conda/envs/pressb/bin/python")
    parser.add_argument("--model-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--seed", type=int, help="First worker seed; defaults to source run.seed")
    parser.add_argument("--startup-timeout", type=float, default=1200.)
    parser.add_argument("--startup-order", choices=("inference-first", "parallel"), default="inference-first",
                        help="Verify the loaded model before starting simulators (default), or overlap service initialization")
    parser.add_argument("--poll-seconds", type=float, default=10.)
    parser.add_argument("--shutdown-timeout", type=float, default=25.)
    parser.add_argument("--expected-task-id")
    parser.add_argument("--expected-weights-sha256")
    parser.add_argument("--max-transitions-per-worker", type=int,
                        help="Diagnostic throughput probe only; partial episodes are never formal evaluation results")
    parser.add_argument("--validate-only", action="store_true", help="Emit and validate CPU configs without starting any node")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    args.output = args.output.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.sim_python = args.sim_python.expanduser().absolute()
    args.model_python = args.model_python.expanduser().absolute()
    # Exclusive directory creation is the launch lock: no partial-run reuse or
    # race between two supervisors targeting the same destination.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(exist_ok=False)
    runner = ShardedEvaluation(args)
    previous = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda *_: setattr(runner, "stopping", True))
        return runner.run()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
