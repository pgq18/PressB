#!/usr/bin/env python3
"""Run paired, frozen XYZ action-filter evaluations with measured trajectories."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pressb.online_rl.rpc import RPCClient

PYTHON = ROOT / ".conda/envs/pressb/bin/python"
OLD = ROOT / "outputs/online_rl_fast_20261003"
TRAIN = ROOT / "outputs/online_rl_residual_on_noise_gamma0995_xyz002_400k_20261008"
CASES = [dict(method="baseline", mode="none", alpha=1.),
         dict(method="xyz_ema_a08", mode="xyz_ema", alpha=.8),
         dict(method="xyz_ema_a06", mode="xyz_ema", alpha=.6),
         dict(method="pose_ema_a08", mode="pose_ema", alpha=.8),
         dict(method="pose_ema_a06", mode="pose_ema", alpha=.6)]


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", default="screen", choices=("screen", "confirm"))
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--cases-file", type=Path)
    args = parser.parse_args()
    cases = read(args.cases_file) if args.cases_file else CASES
    if not isinstance(cases, list) or not cases or len({c["method"] for c in cases}) != len(cases):
        parser.error("Expected a nonempty list of unique candidate methods")
    for case in cases:
        if (not case["method"].replace("_", "").isalnum()
                or case["mode"] not in ("none", "xyz_ema", "pose_ema")
                or type(case["alpha"]) not in (int, float) or not 0 < case["alpha"] <= 1):
            parser.error("Invalid filter candidate")
    config = read(TRAIN / "train_config.json")
    assert config["learner"]["residual_mode"] == "xyz"
    assert config["learner"]["residual_scale"] == [.02] * 3 and config["single_gamma"] == .995
    assert read(TRAIN / "queue_status.json")["state"] == "complete"
    assert read(TRAIN / "train/summary.json")["training_transitions"] == 400000
    for port in (19880, 19891):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                               capture_output=True, text=True, check=True)
    if processes.stdout.strip():
        raise RuntimeError("Existing GPU work is active; wait before starting the smoothing study")
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    children, logs, results, run_map = {}, [], {}, {}

    def save(name, value):
        p = run / name
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        tmp.replace(p)

    def status(stage, state="running", **kwargs):
        save("status.json", dict(state=state, stage=stage, phase=args.phase, results=results,
             updated_at=datetime.now(timezone.utc).isoformat(), **kwargs))

    def launch(name, argv, gpu=None):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   OMNI_KIT_ACCEPT_EULA="YES", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        argv = list(map(str, argv))
        stream = (run / f"{name}.log").open("xb", buffering=0)
        logs.append(stream)
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT)
        children[name] = child
        save(f"{name}_process.json", dict(pid=child.pid, argv=argv,
             starttime_ticks=Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]))
        return child

    def stop(name):
        p = children.get(name)
        if p is None or p.poll() is not None:
            return
        p.terminate()
        try:
            p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=30)

    def interrupt(*_):
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupt)
    try:
        save("plan.json", dict(phase=args.phase, cases=cases, seed=args.seed, episodes_per_method=120,
             floors=list(range(24, 36)), layouts="center plus four corners, two repeats each",
             checkpoint=str(TRAIN / "train/last.pt"), training_run=str(TRAIN),
             noise_checkpoint=str(OLD / "initial_noise_train/last.pt"),
             num_envs=64, single_gamma=.995, weights_frozen=True,
             existing_joint_smoothing_window=3, filter_hz=30,
             candidate_selection="Screen requires success count >= concurrent unfiltered baseline and >=119/120; compare measured joint jerk on common successful cases, then confirm on a different predeclared seed",
             confirmation_seed=(args.seed if args.phase == "confirm" else None),
             source_files={name: identity(ROOT / name) for name in (
                 "src/pressb/online_rl/action_postprocessing.py", "scripts/eval_smoothed_residual.py",
                 "scripts/run_xyz_smoothing_study.py")}))
        status("startup")
        infer = read(OLD / "local_inference_process.json")["argv"]
        launch("inference", infer, gpu=1)
        launch("simulation", [PYTHON, "-u", ROOT / "scripts/serve_rl_recording_simulation.py",
               "--gpu", "0", "--port", "19880", "--num-envs", "64", "--max-seconds", "15", "--camera-resolution", "224",
               "--ik-iterations", "16", "--gpu-dynamics", "--single-gamma", ".995",
               "--output", run / "simulation"])
        deadline = time.monotonic() + 600
        for name, port in (("simulation", 19880), ("inference", 19891)):
            while True:
                if any(children[n].poll() is not None for n in ("simulation", "inference")):
                    raise RuntimeError("A simulation/inference service exited")
                try:
                    health = RPCClient(f"http://127.0.0.1:{port}", timeout=5).call("/health")
                except Exception:
                    health = {}
                if health.get("ready"):
                    if name == "simulation" and (health["num_envs"] != 64 or health["single_gamma"] != .995
                                                or health["smoothing_window"] != 3):
                        raise ValueError("Unexpected simulation control contract")
                    save(f"{name}_health.json", health)
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("Services failed to become ready")
                time.sleep(2)
        frozen_reference = read(TRAIN / "eval/freeze_verification.json")
        for case in cases:
            name = case["method"]
            status(name)
            proc = launch(name, [PYTHON, "-u", ROOT / "scripts/eval_smoothed_residual.py",
                   "--config", TRAIN / "train_config.json", "--noise-checkpoint", OLD / "initial_noise_train/last.pt",
                   "--checkpoint", TRAIN / "train/last.pt", "--output", run / name,
                   "--mode", "eval", "--seed", args.seed, "--eval-episodes", "120", "--device", "cuda:0",
                   "--smoothing-mode", case["mode"], "--smoothing-alpha", case["alpha"]], gpu=0)
            deadline = time.monotonic() + 1200
            while proc.poll() is None:
                if any(children[n].poll() is not None for n in ("simulation", "inference")):
                    raise RuntimeError("A rollout service exited during evaluation")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Evaluation {name} exceeded 20 minutes")
                time.sleep(2)
            if proc.returncode:
                raise RuntimeError(f"Evaluation {name} failed; see {name}.log")
            result = read(run / name / "summary.json")
            if result["state"] != "complete" or result["episodes"] != 120 or result["updates"] != 0:
                raise ValueError("Incomplete or trainable evaluation")
            if read(run / name / "freeze_verification.json") != frozen_reference:
                raise ValueError("Evaluation changed frozen weights")
            results[name] = result
            run_map[result["run_id"]] = dict(method=name, phase=args.phase,
                                              evaluation_directory=str(run / name))
            save("run_map.json", run_map)
            status(name)
        status("complete", state="complete")
    except BaseException as error:
        status("failed", state="failed", error=repr(error))
        raise
    finally:
        for name in reversed(children):
            stop(name)
        for stream in logs:
            stream.close()


if __name__ == "__main__":
    main()
