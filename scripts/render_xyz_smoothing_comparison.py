#!/usr/bin/env python3
"""Render the first centered repeat from two actual 120-task smoothing evals."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".conda/envs/pressb/bin/python"
SCOPE = "First centered repeat for every floor selected from the recorded 120-task confirmation evaluations; no outcome-based selection."


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-run", required=True, type=Path)
    p.add_argument("--smoothed-run", required=True, type=Path)
    p.add_argument("--trajectories", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    sources = dict(residual_xyz=args.baseline_run.resolve(),
                   residual_xyz_smoothed=args.smoothed_run.resolve())
    selected, mapping = [], {}
    entries = [json.loads(line) for line in (args.trajectories / "index.jsonl").read_text().splitlines() if line]
    for method, directory in sources.items():
        summary = read(directory / "summary.json")
        assert summary["state"] == "complete" and summary["episodes"] == 120 and summary["updates"] == 0
        mapping[summary["run_id"]] = method
        candidates = sorted((row for row in entries if row["run_id"] == summary["run_id"]),
                            key=lambda row: row["trajectory_id"])
        assert len(candidates) == 120
        first = {}
        for row in candidates:
            metadata = read(Path(row["directory"]) / "metadata.json")
            layout = metadata["reset_episode"]
            if layout["offset_x_m"] == layout["offset_y_m"] == 0:
                first.setdefault(layout["floor"], row)
        assert set(first) == set(range(24, 36))
        selected.extend(dict(method=method, floor=floor, directory=first[floor]["directory"])
                        for floor in range(24, 36))
    post = read(sources["residual_xyz_smoothed"] / "postprocessing.json")
    assert read(sources["residual_xyz"] / "postprocessing.json")["mode"] == "none"
    assert post["mode"] in ("xyz_ema", "pose_ema")
    assert read(sources["residual_xyz"] / "freeze_verification.json") == read(sources["residual_xyz_smoothed"] / "freeze_verification.json")
    assert read(sources["residual_xyz"] / "manifest.json")["config"]["seed"] == read(sources["residual_xyz_smoothed"] / "manifest.json")["config"]["seed"]
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True)
    if gpu.stdout.strip():
        raise RuntimeError("Finish simulation evaluations before rendering")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        file = output / name
        tmp = file.with_suffix(file.suffix + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        tmp.replace(file)

    def status(stage, state="running", **extra):
        save("status.json", dict(state=state, stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), **extra))

    save("video_plan.json", dict(source_runs={key: str(value) for key, value in sources.items()},
         selected_trajectories=selected, scope=SCOPE, mode=post["mode"], alpha=post["alpha"],
         source_seed=read(sources["residual_xyz"] / "manifest.json")["config"]["seed"]))
    save("run_map.json", mapping)
    labels = dict(residual_xyz="XYZ 残差：未加平滑后处理",
                  residual_xyz_smoothed=("XYZ 平滑" if post["mode"] == "xyz_ema" else "位置＋姿态平滑")
                  + f"（α={post['alpha']:g}）")
    save("method_labels.json", labels)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMNI_KIT_ACCEPT_EULA="YES")
    child = None

    def execute(name, argv):
        nonlocal child
        argv = list(map(str, argv))
        with (output / f"{name}.log").open("xb", buffering=0) as stream:
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
            save(f"{name}_process.json", dict(pid=child.pid, argv=argv))
            code = child.wait(timeout=2400)
            if code:
                raise RuntimeError(f"{name} failed with code {code}")

    def interrupt(*_):
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupt)
    try:
        status("rendering")
        execute("render", [PYTHON, "-u", ROOT / "scripts/render_rl_trajectories.py",
                "--trajectories", *[row["directory"] for row in selected],
                "--run-map", output / "run_map.json", "--output", output / "renders", "--gpu", "0"])
        status("composing")
        execute("compose", [PYTHON, "-u", ROOT / "scripts/compose_rl_comparison_videos.py",
                "--manifest", output / "renders/render_manifest.json", "--output", output / "videos",
                "--methods", "residual_xyz", "residual_xyz_smoothed", "--require-all-center-floors",
                "--method-labels", output / "method_labels.json", "--language", "zh", "--ffmpeg-threads", "2",
                "--scope", SCOPE])
        status("complete", "complete", main_video=str(output / "videos/xyz_vs_smoothed.mp4"))
    except BaseException as error:
        status("failed", "failed", error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=30)


if __name__ == "__main__":
    main()
