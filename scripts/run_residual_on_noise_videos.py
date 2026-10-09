#!/usr/bin/env python3
"""Record all 12 center-panel tasks and make real-time policy comparison videos."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from pressb.online_rl.rpc import RPCClient

OLD = ROOT / "outputs/online_rl_fast_20261003"
DEFAULT_TRAIN = ROOT / "outputs/online_rl_residual_on_noise_400k_20261006"
PYTHON = ROOT / ".conda/envs/pressb/bin/python"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, default=DEFAULT_TRAIN,
                        help="Completed residual-on-frozen-noise experiment providing its final checkpoint")
    parser.add_argument("--comparison-training-run", type=Path,
                        help="Optional pose9 residual run to compare against a half-scale or XYZ residual run")
    args = parser.parse_args()
    train = args.training_run.resolve(strict=True)
    config = json.loads((train / "train_config.json").read_text())
    summary = json.loads((train / "train/summary.json").read_text())
    gamma = config["single_gamma"]
    if type(gamma) not in (int, float) or not math.isfinite(gamma) or not 0 < gamma <= 1:
        parser.error("Training run has an invalid single_gamma")
    if summary["state"] != "complete" or summary["training_transitions"] != 400000:
        parser.error("Expected a completed 400k training run")
    for required in (train / "train/last.pt", train / "eval/manifest.json", train / "eval/freeze_verification.json"):
        if not required.is_file():
            parser.error(f"Missing training/evaluation reference: {required}")
    comparison = args.comparison_training_run.resolve(strict=True) if args.comparison_training_run else None
    xyz = config["learner"].get("residual_mode", "pose9") == "xyz"
    right_method = "residual_xyz" if xyz else "residual_on_noise"
    left_method = "residual_fullscale" if comparison else "base"
    methods = (left_method, right_method)
    video_key = ("pose9_vs_xyz" if comparison else "before_vs_residual_xyz") if xyz else (
        "fullscale_vs_halfscale" if comparison else "before_vs_residual_on_noise")
    training_runs = {right_method: train}
    references = {right_method: str(train / "eval")}
    if comparison:
        previous = json.loads((comparison / "train_config.json").read_text())
        previous_summary = json.loads((comparison / "train/summary.json").read_text())
        if previous_summary["state"] != "complete" or previous_summary["training_transitions"] != 400000:
            parser.error("Comparison must use a completed 400k run")
        if xyz:
            if (previous["learner"].get("residual_mode", "pose9") != "pose9"
                    or previous["learner"]["residual_scale"] != [.03] * 3 + [.1] * 6
                    or config["learner"]["residual_scale"] != [.02] * 3):
                parser.error("XYZ comparison requires original-scale pose9 and XYZ scale=[0.02,0.02,0.02]")
            previous["learner"]["residual_mode"] = "xyz"
        elif config["learner"]["residual_scale"] != [v * .5 for v in previous["learner"]["residual_scale"]]:
            parser.error("Selected run must have exactly half the comparison residual scale")
        previous["learner"]["residual_scale"] = config["learner"]["residual_scale"]
        if previous != config:
            parser.error("Comparison training configurations must differ only in residual scale and (for XYZ) residual mode")
        current_eval = json.loads((train / "eval/manifest.json").read_text())
        comparison_eval = json.loads((comparison / "eval/manifest.json").read_text())
        if current_eval["identities"] != comparison_eval["identities"]:
            parser.error("Comparison simulator, model and frozen-noise contracts differ")
        for filename in ("train/last.pt", "eval/freeze_verification.json", "eval/composition.json"):
            if not (comparison / filename).is_file():
                parser.error(f"Comparison reference missing {filename}")
        training_runs[left_method] = comparison
        references[left_method] = str(comparison / "eval")
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    children = {}
    logs = []
    stage, results = "startup", {}

    def save(name, value):
        path = run / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        tmp.replace(path)

    def status(state="running", **kwargs):
        save("status.json", dict(state=state, stage=stage, results=results,
            updated_at=datetime.now(timezone.utc).isoformat(), **kwargs))

    def environment(gpu=None):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OMNI_KIT_ACCEPT_EULA="YES", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        return env

    def launch(name, argv, gpu=None):
        argv = list(map(str, argv))
        log = (run / f"{name}.log").open("wb", buffering=0)
        logs.append(log)
        process = subprocess.Popen(argv, cwd=ROOT, env=environment(gpu), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT)
        children[name] = process
        record = dict(pid=process.pid, argv=argv,
            starttime_ticks=Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
        save(f"{name}_process.json", record)
        return process

    def execute(name, argv, timeout, gpu=None):
        process = launch(name, argv, gpu)
        if process.wait(timeout=timeout):
            raise RuntimeError(f"{name} exited with code {process.returncode}; see {name}.log")

    def stop(name):
        process = children.get(name)
        if process is None or process.poll() is not None:
            return
        # Popen remains our unreaped direct child; its PID cannot be reused.
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)

    def interrupt(*_):
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupt)
    try:
        status()
        for port in (19880, 19891, 19882):
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", port))
        noise = OLD / "initial_noise_train/last.pt"
        residual = train / "train/last.pt"
        base_config = json.loads((OLD / "online_rl_fast_action_residual.json").read_text())
        base_config["single_gamma"] = gamma
        save("base_config.json", base_config)
        labels = {"base": "强化学习前（原始 VLA-JEPA）",
                  right_method: (f"冻结噪声＋XYZ 残差（scale=0.02）" if xyz else
                                 f"冻结噪声＋动作残差（γ={gamma:g}，400k）")}
        if comparison and xyz:
            labels = {left_method: f"9D 残差：原 scale（γ={gamma:g}）",
                      right_method: f"XYZ 残差：scale=0.02（γ={gamma:g}）"}
        elif comparison:
            labels = {left_method: f"原 scale：冻结噪声＋残差（γ={gamma:g}）",
                      "residual_on_noise": f"半 scale：冻结噪声＋残差（γ={gamma:g}）"}
        save("method_labels.json", labels)
        save("plan.json", dict(methods=list(methods), floors=list(range(24, 36)),
            layout=dict(offset_x_m=0., offset_y_m=0.), seed=20260930, episodes_per_method=12, num_envs=12,
            noise_checkpoint=str(noise), residual_checkpoint=str(residual), single_gamma=gamma,
            training_run=str(train), comparison_training_run=str(comparison) if comparison else None,
            comparison_kind=video_key,
            residual_scales={method: json.loads((directory / "train_config.json").read_text())["learner"]["residual_scale"]
                             for method, directory in training_runs.items()},
            residual_modes={method: json.loads((directory / "train_config.json").read_text())["learner"].get("residual_mode", "pose9")
                            for method, directory in training_runs.items()},
            main_video_key=video_key, evaluation_references=references,
            selection="All predeclared center-panel tasks, retaining both successes and failures",
            recording="Read-only actual states at every 120 Hz tick; unchanged policy camera cadence",
            video="30 fps, global and wrist views, 1x simulation time, ended policies hold final frame",
            evaluation_reference=str(train / "eval"),
            scope="Fresh illustration rollouts with final weights; not footage of the original 120-task eval"))
        infer = json.loads((OLD / "local_inference_process.json").read_text())["argv"]
        launch("inference", infer, gpu=1)
        launch("simulation", [PYTHON, "-u", ROOT / "scripts/serve_rl_recording_simulation.py",
            "--gpu", "0", "--port", "19880", "--num-envs", "12", "--max-seconds", "15",
            "--camera-resolution", "224", "--ik-iterations", "16", "--gpu-dynamics",
            "--single-gamma", str(gamma),
            "--output", run / "simulation"])
        deadline = time.monotonic() + 600
        for name, port in (("simulation", 19880), ("inference", 19891)):
            while True:
                if any(children[key].poll() is not None for key in ("simulation", "inference")):
                    raise RuntimeError("A rollout service exited during startup")
                try:
                    health = RPCClient(f"http://127.0.0.1:{port}", timeout=5).call("/health")
                except Exception:
                    health = {}
                if health.get("ready"):
                    if name == "simulation" and health.get("single_gamma") != gamma:
                        raise ValueError("Recording simulator discount differs from the trained checkpoint")
                    save(f"node_{port}_health.json", health)
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{name} was not ready in time")
                time.sleep(2)
        run_map = {}
        for method in methods:
            stage = f"recording_{method}"
            status()
            common = ["--mode", "eval", "--seed", "20260930", "--eval-episodes", "12",
                "--device", "cuda:0", "--simulation", "http://127.0.0.1:19880",
                "--inference", "http://127.0.0.1:19891", "--output", run / f"{method}_eval"]
            if method == "base":
                argv = [PYTHON, "-u", ROOT / "scripts/run_fast_online_rl.py", "--config",
                        run / "base_config.json", "--method", "base", *common]
            else:
                selected = training_runs[method]
                argv = [PYTHON, "-u", ROOT / "scripts/run_residual_on_noise.py", "--config",
                        selected / "train_config.json", "--noise-checkpoint", noise,
                        "--checkpoint", selected / "train/last.pt", *common]
            execute(f"{method}_eval", argv, 900, gpu=0)
            result = json.loads((run / f"{method}_eval/summary.json").read_text())
            if result["state"] != "complete" or result["episodes"] != 12 or result["updates"] != 0:
                raise RuntimeError(f"{method} did not finish a frozen 12-episode evaluation")
            results[method] = result
            run_map[result["run_id"]] = method
            save("run_map.json", run_map)
        save("recording_status.json", dict(state="complete", results=results))
        stop("simulation")
        stop("inference")
        stage = "rendering"
        status()
        execute("render", [PYTHON, "-u", ROOT / "scripts/render_rl_trajectories.py",
            "--trajectories", run / "simulation/trajectories", "--run-map", run / "run_map.json",
            "--output", run / "renders", "--gpu", "0"], 2400)
        manifest = json.loads((run / "renders/render_manifest.json").read_text())
        if manifest["status"] != "complete" or len(manifest["episodes"]) != 24:
            raise RuntimeError("Rendering did not complete every predeclared trajectory")
        stage = "composing"
        status()
        execute("compose", [PYTHON, "-u", ROOT / "scripts/compose_rl_comparison_videos.py",
            "--manifest", run / "renders/render_manifest.json", "--output", run / "videos",
            "--methods", *methods, "--require-all-center-floors",
            "--method-labels", run / "method_labels.json",
            "--language", "zh", "--ffmpeg-threads", "2"], 2400)
        stage = "complete"
        status("complete", main_video=str(run / f"videos/{video_key}.mp4"))
    except BaseException as error:
        status("failed", error=repr(error))
        raise
    finally:
        for name in reversed(list(children)):
            stop(name)
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
