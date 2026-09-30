#!/usr/bin/env python3
"""Supervise disjoint local Isaac workers; resume with the same GPU list."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="4,6,0,1,2,3,5,7")
    parser.add_argument("--num-envs", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30, help="Synchronized RGB/state/action FPS; must divide120")
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/piper_elevator_raw_panel_randomized_30hz")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dataset_panel_randomized.json")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "outputs/edge_feedback/scene.usda")
    parser.add_argument("--reset-renderer-accumulation", action="store_true", default=True,
                        help="Compatibility flag: renderer history is always reset at lamp transitions")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    if (len(set(gpus)) != len(gpus) or min(gpus) < 0 or min(args.num_envs, args.episodes_per_task, args.fps) < 1
            or 120 % args.fps):
        parser.error("Require unique nonnegative GPUs and positive counts")
    args.output = args.output.resolve()
    args.config, args.snapshot = args.config.resolve(), args.snapshot.resolve()
    if not args.config.is_file() or not args.snapshot.is_file():
        parser.error("Config and scene snapshot must exist; build the scene with scripts/run.sh first")
    args.output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = ROOT / "logs" / f"collection_{run_id}"
    log_dir.mkdir()
    sources = ["scripts/collect_parallel.py", "scripts/collect_dataset.py", "scripts/collect_dataset.sh",
               "src/pressb/dataset_planning.py", "src/pressb/dataset_scene.py", "src/pressb/scene.py",
               "src/pressb/planning.py", "src/pressb/panel_randomization.py", "src/pressb/panel_metadata.py",
               "configs/dataset_panel_randomized.json"]
    manifest = dict(started_at=run_id, requested_episodes=12 * args.episodes_per_task,
                    episodes_per_task=args.episodes_per_task, seed=args.seed, raw=str(args.output), fps=args.fps,
                    config=str(args.config), snapshot=str(args.snapshot),
                    config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                    snapshot_sha256=hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
                    workers=[dict(worker_index=i, gpu=gpu, num_envs=args.num_envs) for i, gpu in enumerate(gpus)],
                    files={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources})
    processes, handles = [], []
    try:
        for worker, gpu in enumerate(gpus):
            command = ["bash", str(ROOT / "scripts/collect_dataset.sh"), "--output", str(args.output),
                       "--config", str(args.config), "--snapshot", str(args.snapshot),
                       "--gpu", str(gpu), "--num-envs", str(args.num_envs),
                       "--fps", str(args.fps),
                       "--episodes-per-task", str(args.episodes_per_task), "--seed", str(args.seed),
                       "--workers", str(len(gpus)), "--worker-index", str(worker)]
            if args.reset_renderer_accumulation:
                command.append("--reset-renderer-accumulation")
            handle = (log_dir / f"worker_{worker:02d}.log").open("wb")
            handles.append(handle)
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            processes.append(process)
            manifest["workers"][worker].update(pid=process.pid, command=command, log=str(handle.name))
        (log_dir / "run.json").write_text(json.dumps(manifest, indent=2))
        (args.output / "collection_run.json").write_text(json.dumps(manifest, indent=2))
        while True:
            results = [process.poll() for process in processes]
            count = sum(path.is_dir() for path in args.output.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))
            print(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), committed=count,
                                  requested=manifest["requested_episodes"], exits=results)), flush=True)
            if any(code not in (None, 0) for code in results):
                raise RuntimeError(f"Worker failed; inspect {log_dir}: {results}")
            if all(code == 0 for code in results):
                if count != manifest["requested_episodes"]:
                    raise RuntimeError(f"Incorrect committed count: {count}")
                break
            time.sleep(20)
    finally:
        # Only signal child process groups created by this invocation.
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
