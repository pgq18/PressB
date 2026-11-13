#!/usr/bin/env python3
"""Wait for a verified SFT archive, then evaluate its best model in Isaac.

Run with .conda/envs/vlajepa-inference/bin/python. No SSH or checkpoint copy
occurs here. The archive queue remains the sole model transfer owner.

Formal evaluation:
  python scripts/run_cube_stack_sft_eval.py --archive-receipt <queue>/archive_workers/cube_stack/cube_stack.archive.json --output outputs/cube_stack_sft_eval

CPU preflight only: append --validate-only (waits for the same archive).
One-episode diagnostic with an already archived, possibly interrupted run:
  python scripts/run_cube_stack_sft_eval.py --smoke-run <local-run> --output outputs/cube_stack_sft_smoke

Smoke results are explicitly marked diagnostic and never reused as the formal
100-episode result. Use a fresh output directory after an interrupted eval;
physics RPCs and partially evaluated episodes are never replayed automatically.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
INSTRUCTION = "Pick up the red cube and place it on top of the blue cube."


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_ino, before.st_mtime_ns, before.st_size) != (after.st_ino, after.st_mtime_ns, after.st_size):
        raise RuntimeError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def worker_receipt(archive):
    # Queue uses <queue>/cube_stack.archive_worker.json next to archive_workers/.
    if archive.parent.name == "cube_stack" and archive.parent.parent.name == "archive_workers":
        return archive.parents[2] / "cube_stack.archive_worker.json"
    return None


def check_archive_worker(path):
    row = read_json(path) if path is not None else None
    if row is None:
        return
    if row.get("status") == "failed" or row.get("exit_code", 0) != 0:
        raise RuntimeError(f"Archive worker failed: {path}; log={row.get('log')}; error={row.get('error')}")
    if row.get("status") == "running":
        try:
            fields = Path(f"/proc/{int(row['supervisor_pid'])}/stat").read_text().rsplit(")", 1)[1].split()
            alive = fields[0] != "Z" and fields[19] == row.get("start_ticks")
        except (FileNotFoundError, KeyError, ValueError, IndexError):
            alive = False
        if not alive:
            # Avoid mistaking the final atomic receipt update for a lost worker.
            reread = read_json(path)
            if reread.get("status") != "completed":
                raise RuntimeError(f"Archive worker disappeared without completion: {path}")


def verify_archive(record):
    if record.get("task") != "cube_stack" or not record.get("verified_unix"):
        raise ValueError("Expected a completely verified cube_stack archive receipt")
    manifest = record.get("manifest", {})
    if record.get("manifest_sha256") != fingerprint(manifest):
        raise ValueError("Archive manifest digest differs from its receipt")
    root = Path(record["archive_path"]).resolve(strict=True)
    files, links = manifest.get("files"), manifest.get("symlinks")
    if not isinstance(files, dict) or not isinstance(links, dict) or not files:
        raise ValueError("Archive receipt must cover its complete file tree")
    # Check all files still exist and match size. The queue already hashed the
    # entire archive; here rehash metadata and selected best weights below.
    for relative, expected in files.items():
        path = root / relative
        if Path(relative).is_absolute() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"Unsafe archive entry: {relative}")
        if path.is_symlink() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Archive file size/type changed: {relative}")
        if path.stat().st_size <= 16 * 1024**2 and sha256(path) != expected["sha256"]:
            raise ValueError(f"Archive metadata changed: {relative}")
    for relative, target in links.items():
        path = root / relative
        if (Path(relative).is_absolute() or not path.is_symlink() or os.readlink(path) != target
                or Path(target).is_absolute() or not path.resolve(strict=True).is_relative_to(root)):
            raise ValueError(f"Archive symlink changed or escapes root: {relative}")
    return root


def inspect_checkpoint(run, archive=None):
    from pressb.models.vlajepa_joint import read_checkpoint_metadata

    metadata = read_checkpoint_metadata(run)
    if (metadata["schema"].get("schema") != "vlajepa.joint7.v1"
            or metadata["schema"].get("task_id") != "cube_stack"
            or metadata["allowed_instructions"] != [INSTRUCTION]
            or metadata["action_dim"] != 7
            or metadata["config"]["datasets"]["vla_data"]["resolution_size"] != 224
            or metadata["schema"].get("fps") != 30):
        raise ValueError("Expected the cube_stack joint7 / up,wrist / 224px / 30Hz checkpoint")
    digest = sha256(metadata["model_path"])
    if metadata["expected_sha256"] not in (None, digest):
        raise ValueError("Best model SHA256 differs from published checkpoint metadata")
    if archive is not None:
        relative = metadata["model_path"].resolve().relative_to(Path(run).resolve()).as_posix()
        entry = archive["manifest"]["files"].get(relative)
        if entry is None or entry["sha256"] != digest:
            raise ValueError("Best model SHA256 differs from verified full archive")
    return dict(run=str(metadata["run"]), checkpoint=str(metadata["checkpoint"]),
                model_path=str(metadata["model_path"]), model_sha256=digest,
                step=metadata["manifest"]["step"], run_id=metadata["config"]["run_id"],
                dataset_sha256=metadata["schema"]["dataset_sha256"],
                training_fingerprint=metadata["schema"]["training_fingerprint"])


def prepare_config(args, checkpoint):
    from pressb.experiments.config import load_experiment
    from pressb.runtime.trainer import TrainerConfig
    from pressb.tasks.cube_stack.configuration import load_config

    config = read_json(args.config)
    config = json.loads(json.dumps(config).replace("${project_root}", str(ROOT)))
    config["model"]["config"].update(run=checkpoint["run"], checkpoint=checkpoint["checkpoint"],
                                      device=f"cuda:{args.inference_gpu}")
    task = config["task"]["config"]
    episodes, num_envs = (1, 1) if args.smoke_run else (100, 8)
    task.update(num_envs=num_envs, gpu=args.sim_gpu, log_dir=str(args.output / "simulation"))
    config["nodes"]["inference"]["service"]["batch_size"] = num_envs
    config["nodes"]["inference"]["server"]["port"] = args.inference_port
    config["nodes"]["simulation"]["server"]["port"] = args.simulation_port
    config["run"]["seed"] = args.seed
    config["run"]["rollout"].update(eval_episodes=episodes, num_envs=num_envs)
    config["run"]["output_dir"] = str(args.output / "run")
    task_config = load_config(deepcopy(task))
    if task_config["control_hz"] != 30 or task_config["horizon"] != 7:
        raise ValueError("Evaluation requires 30Hz and 7-action chunks")
    for key in ("base_vlm", "base_encoder"):
        if not (Path(config["model"]["config"][key]) / "config.json").is_file():
            raise FileNotFoundError(f"Missing local architecture config: {key}")
    path = args.output / "experiment.json"
    write_json(path, config)
    compiled = load_experiment(path)
    TrainerConfig(**deepcopy(compiled["nodes"]["trainer"]))
    return path, compiled, episodes


def gpu_snapshot():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=20)
    rows = {}
    for line in result.stdout.strip().splitlines():
        index, name, total, used, free, util = [item.strip() for item in line.split(",")]
        rows[int(index)] = dict(name=name, total_mib=int(total), used_mib=int(used), free_mib=int(free), utilization=int(util))
    return rows


def port_available(port):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def summarize(run, expected):
    status = read_json(run / "status.json", {})
    path = run / "episodes.jsonl"
    episodes = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
    if status.get("state") != "completed" or len(episodes) != expected or status.get("episodes") != expected:
        raise RuntimeError(f"Eval did not complete exactly {expected} episodes: {status.get('state')}, {len(episodes)} rows")
    if len({(row["env_id"], row["episode_id"]) for row in episodes}) != expected:
        raise ValueError("Repeated completed episode identity")
    for row in episodes:
        physical = row.get("info", {}).get("physical_outcome", {})
        if type(row.get("success")) is not bool or physical.get("success") is not row["success"]:
            raise ValueError("Episode success disagrees with strict physical stacking outcome")
    successes = sum(row["success"] for row in episodes)
    if successes != status.get("successes"):
        raise ValueError("Summary success count disagrees with episode records")
    rate = successes / expected
    z = 1.959963984540054
    centre = (rate + z*z/(2*expected)) / (1+z*z/expected)
    radius = z*math.sqrt(rate*(1-rate)/expected+z*z/(4*expected**2)) / (1+z*z/expected)
    return dict(episodes=expected, successes=successes, failures=expected-successes,
                success_rate=rate, success_rate_wilson95=[centre-radius, centre+radius],
                termination_reasons=dict(Counter(str(row["info"].get("terminal_reason", row["info"].get("termination_reason", "unspecified"))) for row in episodes)),
                runner_status=status)


class Evaluation:
    def __init__(self, args):
        self.args = args
        self.children = {}
        self.logs = []
        self.stopping = False
        self.state = dict(status="starting", pid=os.getpid(), started_unix=time.time(),
                          mode="diagnostic_one_episode" if args.smoke_run else "formal_100_episodes",
                          archive_receipt=str(args.archive_receipt) if args.archive_receipt else None)

    def update(self, status, **extra):
        self.state.update(status=status, updated_unix=time.time(), **extra)
        write_json(self.args.output / "status.json", self.state)
        print(json.dumps(dict(status=status, **extra), default=str, allow_nan=False), flush=True)

    def interrupted(self):
        if self.stopping:
            raise InterruptedError("Evaluation supervisor received SIGINT/SIGTERM")

    def pause(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.interrupted()
            time.sleep(min(1, max(0, end-time.monotonic())))

    def await_archive(self):
        if self.args.smoke_run:
            return self.args.smoke_run.resolve(strict=True), None
        receipt = self.args.archive_receipt
        while True:
            self.interrupted()
            check_archive_worker(worker_receipt(receipt))
            row = read_json(receipt)
            if row and row.get("verified_unix"):
                self.update("verifying_archive", archive_path=row.get("archive_path"))
                return verify_archive(row), row
            worker = worker_receipt(receipt)
            if worker is not None:
                queue_path = worker.parent / "status.json"
                queue = read_json(queue_path, {})
                if queue.get("status") in ("failed", "superseded"):
                    raise RuntimeError(f"SFT/archive queue is {queue['status']} before verified archive: "
                                       f"{queue_path}; error={queue.get('error')}")
            self.update("waiting_archive")
            self.pause(self.args.poll_seconds)

    def launch(self, name, python, config):
        command = [str(python), "-u", str(ROOT / "scripts/run_experiment.py"), name,
                   "--config", str(config), "--resolved-dir", str(self.args.output / "resolved" / name)]
        log_path = self.args.output / f"{name}.log"
        log = log_path.open("a")
        self.logs.append(log)
        env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        # Host GPU IDs in this script refer to the unrestricted local machine.
        env.pop("CUDA_VISIBLE_DEVICES", None)
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.children[name] = child
        self.state.setdefault("processes", {})[name] = dict(pid=child.pid, command=command, log=str(log_path))
        self.update("starting_nodes" if name != "eval" else "evaluating")

    def check_children(self, include_eval=False):
        self.interrupted()
        for name, child in self.children.items():
            if name == "eval" and not include_eval:
                continue
            code = child.poll()
            if code is not None:
                raise RuntimeError(f"Owned {name} node exited with code {code}; see {self.args.output / (name + '.log')}")

    def await_health(self, compiled):
        from pressb.runtime.client import NodeClient
        pending = {name: compiled["nodes"]["trainer"]["endpoints"][name] for name in ("simulation", "inference")}
        deadline = time.monotonic() + self.args.startup_timeout
        errors = {}
        while pending:
            self.check_children()
            for name, endpoint in list(pending.items()):
                client = NodeClient(endpoint, timeout=3, token=os.environ.get("PRESSB_RL_TOKEN"))
                try:
                    health = client.call("/health")
                    if health.get("schema_version") != 2 or health.get("node_type") != name:
                        raise ValueError(f"Wrong service metadata at {endpoint}")
                    write_json(self.args.output / f"{name}.health.json", health)
                    del pending[name]
                except Exception as error:
                    errors[name] = str(error)
                finally:
                    client.close()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Services not ready: {pending}; last errors={errors}")
            if pending:
                self.pause(3)

    def cleanup(self):
        # Every group was created by this supervisor; never touch unrelated jobs.
        for name in ("eval", "simulation", "inference"):
            child = self.children.get(name)
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 25
        for child in self.children.values():
            try:
                child.wait(timeout=max(.1, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=10)
        for log in self.logs:
            log.close()

    def run(self):
        args = self.args
        try:
            run, archive = self.await_archive()
            self.update("verifying_best_checkpoint", run=str(run))
            checkpoint = inspect_checkpoint(run, archive)
            self.state["checkpoint"] = checkpoint
            self.state["archive_manifest_sha256"] = archive.get("manifest_sha256") if archive else None
            path, compiled, expected = prepare_config(args, checkpoint)
            self.state.update(experiment_sha256=sha256(path), seed=args.seed, expected_episodes=expected)
            if args.validate_only:
                self.update("validated", gpu_processes_started=False)
                return 0
            if (args.output / "run").exists():
                raise FileExistsError("Runner output exists; use a fresh output directory (no automatic replay)")
            for python in (args.model_python, args.sim_python):
                if not python.is_file():
                    raise FileNotFoundError(python)
            while True:
                hardware = gpu_snapshot()
                if any(gpu not in hardware for gpu in (args.sim_gpu, args.inference_gpu)):
                    raise ValueError("Requested GPUs are absent")
                if all(hardware[gpu]["free_mib"] >= args.min_free_mib for gpu in (args.sim_gpu, args.inference_gpu)):
                    break
                self.update("waiting_gpus", hardware=hardware)
                self.pause(args.poll_seconds)
            write_json(args.output / "hardware_before.json", hardware)
            for port in (args.simulation_port, args.inference_port):
                port_available(port)
            self.launch("simulation", args.sim_python, path)
            self.launch("inference", args.model_python, path)
            self.await_health(compiled)
            self.launch("eval", args.model_python, path)
            next_report = 0.
            while self.children["eval"].poll() is None:
                self.check_children()
                if time.monotonic() >= next_report:
                    self.update("evaluating", progress=read_json(args.output / "run/status.json", {}))
                    next_report = time.monotonic() + args.poll_seconds
                self.pause(1)
            self.check_children()
            if self.children["eval"].returncode != 0:
                raise RuntimeError(f"Eval runner exited with code {self.children['eval'].returncode}; see eval.log")
            summary = summarize(args.output / "run", expected)
            result = dict(summary, checkpoint=checkpoint, seed=args.seed, mode=self.state["mode"],
                          archive_manifest_sha256=self.state["archive_manifest_sha256"],
                          experiment_sha256=self.state["experiment_sha256"], finished_unix=time.time())
            write_json(args.output / "result.json", result)
            self.update("completed", successes=summary["successes"], episodes=expected,
                        success_rate=summary["success_rate"], result=str(args.output / "result.json"))
            return 0
        except BaseException as error:
            self.update("stopped" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                        error_type=type(error).__name__, error=str(error))
            raise
        finally:
            self.cleanup()
            self.state["children_cleaned_up"] = True
            write_json(args.output / "status.json", self.state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive-receipt", type=Path)
    source.add_argument("--smoke-run", type=Path, help="Already local run: exactly one diagnostic episode, no formal result")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/cube_stack_vlajepa_sft_eval.json")
    parser.add_argument("--model-python", type=Path, default=ROOT / ".conda/envs/vlajepa-inference/bin/python")
    parser.add_argument("--sim-python", type=Path, default=ROOT / ".conda/envs/pressb/bin/python")
    parser.add_argument("--sim-gpu", type=int, default=0)
    parser.add_argument("--inference-gpu", type=int, default=1)
    parser.add_argument("--inference-port", type=int, default=18653)
    parser.add_argument("--simulation-port", type=int, default=18623)
    parser.add_argument("--seed", type=int, default=2026111101)
    parser.add_argument("--min-free-mib", type=int, default=24000)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.sim_gpu == args.inference_gpu or min(args.sim_gpu, args.inference_gpu) < 0:
        parser.error("Use two distinct local GPUs")
    if args.inference_port == args.simulation_port or any(not 1024 <= port <= 65535 for port in (args.inference_port, args.simulation_port)):
        parser.error("Use distinct unprivileged ports")
    if not (0 <= args.seed < 2**32) or args.min_free_mib <= 0 or any(not math.isfinite(v) or v <= 0 for v in (args.poll_seconds, args.startup_timeout)):
        parser.error("Invalid seed, available-memory threshold or timing")
    args.output = args.output.expanduser().resolve()
    args.config = args.config.expanduser().absolute()
    if args.archive_receipt:
        args.archive_receipt = args.archive_receipt.expanduser().absolute()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / "status.json").exists():
            parser.error("Output already contains supervisor state; select a fresh directory")
        runner = Evaluation(args)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: setattr(runner, "stopping", True))
        return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
