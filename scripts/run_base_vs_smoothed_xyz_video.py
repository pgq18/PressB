#!/usr/bin/env python3
"""Record matched base/XYZ-smoothed rollouts and show two tasks simultaneously."""
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
METHODS = ("base", "residual_xyz_smoothed")


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--floors", nargs=2, type=int, default=[25, 31])
    args = parser.parse_args()
    if len(set(args.floors)) != 2 or any(f not in range(24, 36) for f in args.floors):
        parser.error("Select two different floors from 24 through 35")
    config = read(TRAIN / "train_config.json")
    assert config["learner"]["residual_mode"] == "xyz"
    assert config["learner"]["residual_scale"] == [.02] * 3 and config["single_gamma"] == .995
    assert read(TRAIN / "train/summary.json")["training_transitions"] == 400000
    for port in (19880, 19891):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         check=True, capture_output=True, text=True)
    if gpu.stdout.strip():
        raise RuntimeError("Existing GPU work is active")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    children, streams, results, mapping = {}, [], {}, {}

    def save(name, value):
        p = output / name
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        tmp.replace(p)

    def status(stage, state="running", **extra):
        save("status.json", dict(state=state, stage=stage, results=results,
             updated_at=datetime.now(timezone.utc).isoformat(), **extra))

    def launch(name, argv, gpu=None):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OMNI_KIT_ACCEPT_EULA="YES", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        argv = list(map(str, argv))
        stream = (output / f"{name}.log").open("xb", buffering=0)
        streams.append(stream)
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT)
        children[name] = child
        save(f"{name}_process.json", dict(pid=child.pid, argv=argv))
        return child

    def execute(name, argv, timeout=1200, gpu=None):
        child = launch(name, argv, gpu)
        if child.wait(timeout=timeout):
            raise RuntimeError(f"{name} failed with code {child.returncode}")

    def stop(name):
        child = children.get(name)
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=30)

    def interrupt(*_):
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupt)
    noise = OLD / "initial_noise_train/last.pt"
    plan = dict(methods=list(METHODS), floors=args.floors, recorded_floors=list(range(24, 36)),
                layout=dict(offset_x_m=0., offset_y_m=0.), seed=20260930,
                episodes_per_method=12, num_envs=12, single_gamma=.995,
                training_run=str(TRAIN), evaluation_reference=str(TRAIN / "eval"),
                noise_checkpoint=str(noise), residual_checkpoint=str(TRAIN / "train/last.pt"),
                residual_mode="xyz", residual_scale=[.02] * 3,
                smoothing_mode="xyz_ema", smoothing_alpha=.8,
                weights_updated=False, extra_smoothing_is_experimental=True,
                selection="User selected floors before recording; first centered episode, retain all outcomes",
                scope="Fresh matched illustration rollouts, not the original 120-task evaluation",
                playback="Both tasks simultaneously at 1x simulation time; each terminal frame held and labeled",
                source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                    "scripts/run_base_vs_smoothed_xyz_video.py", "scripts/eval_smoothed_residual.py",
                    "src/pressb/online_rl/action_postprocessing.py")})
    try:
        save("plan.json", plan)
        base_config = read(OLD / "online_rl_fast_action_residual.json")
        base_config["single_gamma"] = .995
        save("base_config.json", base_config)
        status("startup")
        launch("inference", read(OLD / "local_inference_process.json")["argv"], gpu=1)
        launch("simulation", [PYTHON, "-u", ROOT / "scripts/serve_rl_recording_simulation.py",
               "--gpu", "0", "--port", "19880", "--num-envs", "12", "--max-seconds", "15",
               "--camera-resolution", "224", "--ik-iterations", "16", "--gpu-dynamics",
               "--single-gamma", ".995", "--output", output / "simulation"])
        deadline = time.monotonic() + 600
        for name, port in (("simulation", 19880), ("inference", 19891)):
            while True:
                if any(children[n].poll() is not None for n in ("simulation", "inference")):
                    raise RuntimeError("A rollout service exited during startup")
                try:
                    health = RPCClient(f"http://127.0.0.1:{port}", timeout=5).call("/health")
                except Exception:
                    health = {}
                if health.get("ready"):
                    save(f"node_{port}_health.json", health)
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{name} failed to become ready")
                time.sleep(2)
        for method in METHODS:
            status(f"recording_{method}")
            common = ["--mode", "eval", "--seed", "20260930", "--eval-episodes", "12",
                      "--device", "cuda:0", "--simulation", "http://127.0.0.1:19880",
                      "--inference", "http://127.0.0.1:19891", "--output", output / f"{method}_eval"]
            if method == "base":
                argv = [PYTHON, "-u", ROOT / "scripts/run_fast_online_rl.py", "--config",
                        output / "base_config.json", "--method", "base", *common]
            else:
                argv = [PYTHON, "-u", ROOT / "scripts/eval_smoothed_residual.py", "--config",
                        TRAIN / "train_config.json", "--noise-checkpoint", noise,
                        "--checkpoint", TRAIN / "train/last.pt", "--smoothing-mode", "xyz_ema",
                        "--smoothing-alpha", ".8", *common]
            execute(f"{method}_eval", argv, gpu=0)
            result = read(output / f"{method}_eval/summary.json")
            if result["state"] != "complete" or result["episodes"] != 12 or result["updates"] != 0:
                raise ValueError("Expected a complete frozen 12-task evaluation")
            results[method] = result
            mapping[result["run_id"]] = method
            save("run_map.json", mapping)
        reference = read(TRAIN / "eval/manifest.json")["identities"]
        right = output / "residual_xyz_smoothed_eval"
        assert read(right / "freeze_verification.json") == read(TRAIN / "eval/freeze_verification.json")
        assert read(right / "manifest.json")["identities"] == reference
        base = read(output / "base_eval/manifest.json")["identities"]
        assert base["inference"] == reference["inference"] and base["simulation"] == reference["simulation"]
        selected = []
        entries = [json.loads(line) for line in (output / "simulation/trajectories/index.jsonl").read_text().splitlines() if line]
        for method in METHODS:
            for floor in args.floors:
                matches = []
                for entry in entries:
                    if mapping[entry["run_id"]] != method:
                        continue
                    meta = read(Path(entry["directory"]) / "metadata.json")
                    reset = meta["reset_episode"]
                    if reset == dict(floor=floor, offset_x_m=0., offset_y_m=0.):
                        matches.append(entry)
                assert len(matches) == 1, (method, floor, len(matches))
                selected.append(dict(method=method, floor=floor, directory=matches[0]["directory"]))
        save("video_plan.json", {**plan, "source_runs": {m: str(output / f"{m}_eval") for m in METHODS},
             "selected_trajectories": selected})
        save("recording_status.json", dict(state="complete", results=results))
        stop("simulation")
        stop("inference")
        status("rendering")
        execute("render", [PYTHON, "-u", ROOT / "scripts/render_rl_trajectories.py",
                "--trajectories", *[row["directory"] for row in selected],
                "--run-map", output / "run_map.json", "--output", output / "renders", "--gpu", "0"], 1800)
        assert len(read(output / "renders/render_manifest.json")["episodes"]) == 4
        status("composing")
        execute("compose", [PYTHON, "-u", ROOT / "scripts/compose_two_button_comparison.py",
                "--manifest", output / "renders/render_manifest.json", "--output", output / "videos",
                "--floors", *args.floors, "--methods", *METHODS], 1800)
        status("complete", "complete", main_video=str(output / "videos/two_button_comparison.mp4"))
    except BaseException as error:
        status("failed", "failed", error=repr(error))
        raise
    finally:
        for name in reversed(children):
            stop(name)
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
