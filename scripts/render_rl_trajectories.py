#!/usr/bin/env python3
"""Render measured RL trajectories without re-executing their controllers.

The input recorder saves actual full robot DOFs and button rigid-body states at
120 Hz. This tool restores those states into one Isaac environment and renders
global/wrist RGB at 30 Hz, including the exact terminal state. No physics step is
allowed after the renderer's initial scene/home initialization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def discover(roots, run_map):
    episodes = []
    seen = set()
    for root in roots:
        root = root.resolve()
        candidates = [root / "metadata.json"] if (root / "physics.npz").is_file() else sorted(root.rglob("metadata.json"))
        for path in candidates:
            if not (path.parent / "physics.npz").is_file() or path.resolve() in seen:
                continue
            seen.add(path.resolve())
            meta = json.loads(path.read_text())
            if meta.get("schema_version") != 1 or meta.get("status") != "complete":
                raise ValueError(f"Expected a complete schema-1 trajectory: {path}")
            method = run_map.get(str(meta["run_id"]), meta.get("method"))
            if method not in ("base", "action_residual", "initial_noise", "combined", "residual_on_noise", "residual_fullscale", "residual_xyz", "residual_xyz_smoothed"):
                raise ValueError(f"Unknown method for run {meta['run_id']}; provide --run-map: {path}")
            episodes.append(dict(path=path.parent, metadata=meta, method=method))
    if not episodes:
        raise ValueError("No complete measured trajectories were found")
    return sorted(episodes, key=lambda item: (item["method"], item["metadata"]["reset_episode"]["floor"], str(item["path"])))


def load_trajectory(episode):
    meta = episode["metadata"]
    expected_identity = meta.get("physics_file", {})
    if expected_identity.get("sha256") != identity(episode["path"] / "physics.npz")["sha256"]:
        raise ValueError(f"Recorded trajectory checksum mismatch: {episode['path']}")
    if meta.get("button_floors") != list(range(24, 36)):
        raise ValueError("Recorded button order differs from the rendering scene")
    with np.load(episode["path"] / "physics.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    n = int(meta["samples"])
    joints = list(meta["joint_names"])
    shapes = {"physics_index": (n,), "sim_time": (n,), "world_time": (n,),
              "q_actual": (n, len(joints)), "qd_actual": (n, len(joints)),
              "button_position_world": (n, 12, 3), "button_orientation_wxyz": (n, 12, 4),
              "button_velocity_world": (n, 12, 6)}
    for name, shape in shapes.items():
        if name not in data or data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError(f"Invalid {name} in {episode['path']}: expected {shape}")
    ticks = data["physics_index"]
    if n < 2 or not np.array_equal(ticks, np.arange(n)):
        raise ValueError("Trajectories must include every physical tick from initial tick zero")
    if not np.allclose(data["sim_time"], ticks / 120., rtol=0, atol=1e-9):
        raise ValueError("Trajectory timestamps are inconsistent with 120 Hz physics")
    if int(meta["info"]["physics_index"]) != n - 1:
        raise ValueError("The trajectory does not end at its reported terminal physics tick")
    if abs(float(meta["info"]["sim_seconds"]) - (n - 1) / 120.) > 1e-9:
        raise ValueError("The reported episode duration differs from the measured terminal tick")
    if not np.allclose(np.diff(data["world_time"]), 1 / 120., rtol=0, atol=1e-7):
        raise ValueError("Recorded world timestamps do not advance by exactly one physics tick")
    if abs(float(meta["physics_dt"]) - 1 / 120.) > 1e-12:
        raise ValueError("Only the trained 120 Hz physics contract is supported")
    indices = list(range(0, n, 4))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    return data, indices


def video_probe(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames,duration,codec_name,pix_fmt",
        "-of", "json", str(path)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)["streams"][0]


def render_episode(backend, episode, output):
    from PIL import Image
    from collect_dataset import VideoPipe, validate_encoded_video
    from pressb.dataset_scene import set_light, set_panel_offset

    meta, method = episode["metadata"], episode["method"]
    reset, info = meta["reset_episode"], meta["info"]
    source = identity(episode["path"] / "physics.npz")
    source_meta = identity(episode["path"] / "metadata.json")
    name = f"floor_{int(reset['floor']):02d}_{source_meta['sha256'][:12]}"
    directory = output / method / name
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "source_metadata.json", meta)
    data, indices = load_trajectory(episode)
    actual_names = list(backend.arms.view.shared_metatype.dof_names)
    recorded_names = list(meta["joint_names"])
    if len(set(recorded_names)) != len(recorded_names) or set(actual_names) != set(recorded_names):
        raise ValueError("Recorded and renderer robot DOF names differ")
    order = [recorded_names.index(name) for name in actual_names]
    env = backend.envs[0]
    offset = np.asarray(meta["env_offset_m"], dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Invalid recorded environment offset")
    time_before = float(backend.world.current_time)
    set_panel_offset(backend.world, env, reset["offset_y_m"], reset["offset_x_m"])
    row = backend.arm_rows[[0]]
    button_rows = backend.button_rows[0]
    events = sorted(info.get("events", []), key=lambda event: event["physics_index"])
    event_cursor = 0
    lit = set()
    frame_rows = []
    checks = dict(max_joint_position_restore_error=0., max_joint_velocity_restore_error=0.,
                  max_button_position_restore_error_m=0., max_button_orientation_restore_error=0.,
                  max_button_velocity_restore_error=0.)
    pipes = {}
    try:
        pipes = {camera: VideoPipe(directory / f"{camera}.mp4", 30) for camera in ("global", "wrist")}
        for frame, sample in enumerate(indices):
            q = data["q_actual"][sample, order][None]
            qd = data["qd_actual"][sample, order][None]
            positions = data["button_position_world"][sample] - offset + np.asarray(env.offset)
            orientations = data["button_orientation_wxyz"][sample]
            velocities = data["button_velocity_world"][sample]
            backend.arms.set_joint_positions(q, indices=row)
            backend.arms.set_joint_velocities(qd, indices=row)
            backend.arms.set_joint_position_targets(q, indices=row)
            backend.button_view.set_world_poses(positions, orientations, indices=button_rows)
            backend.button_view.set_velocities(velocities, indices=button_rows)
            changed = frame == 0
            while event_cursor < len(events) and events[event_cursor]["physics_index"] <= sample:
                event = events[event_cursor]
                if event["type"] not in ("pressed", "released"):
                    raise ValueError(f"Unknown measured lamp event: {event}")
                floor, on = int(event["floor"]), event["type"] == "pressed"
                set_light(env, floor, on)
                lit.add(floor) if on else lit.discard(floor)
                changed = True
                event_cursor += 1
            tiles = backend._capture(changed=changed)
            restored_q, restored_qd = backend._joint_arrays()
            restored_p, restored_o = backend.button_view.get_world_poses()
            restored_v = backend.button_view.get_velocities()
            errors = dict(max_joint_position_restore_error=float(np.max(np.abs(restored_q[0] - q[0]))),
                max_joint_velocity_restore_error=float(np.max(np.abs(restored_qd[0] - qd[0]))),
                max_button_position_restore_error_m=float(np.max(np.abs(restored_p[button_rows] - positions))),
                max_button_orientation_restore_error=float(np.max(np.abs(restored_o[button_rows] - orientations))),
                max_button_velocity_restore_error=float(np.max(np.abs(restored_v[button_rows] - velocities))))
            if any(value > 2e-6 for value in errors.values()):
                raise RuntimeError(f"Measured state failed to restore at frame {frame}: {errors}")
            for key, value in errors.items():
                checks[key] = max(checks[key], value)
            if float(backend.world.current_time) != time_before:
                raise RuntimeError("Offline rendering advanced the physics clock")
            frame_rows.append(dict(frame=frame, video_time=frame / 30., physics_index=int(sample),
                sim_time=float(data["sim_time"][sample]), terminal=sample == len(data["sim_time"]) - 1,
                lit_floors=sorted(lit)))
            for camera, tile in (("global", tiles[1]), ("wrist", tiles[0])):
                pipes[camera].add(tile)
                if frame == 0:
                    Image.fromarray(tile).save(directory / f"start_{camera}.jpg", quality=95)
                if frame == len(indices) - 1:
                    Image.fromarray(tile).save(directory / f"end_{camera}.jpg", quality=95)
            if frame % 150 == 0:
                print(json.dumps(dict(event="render_progress", method=method, floor=reset["floor"],
                    frame=frame, total_frames=len(indices))), flush=True)
    finally:
        errors = []
        for pipe in pipes.values():
            try:
                pipe.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]
    video_metadata = {}
    for camera in ("global", "wrist"):
        path = directory / f"{camera}.mp4"
        validate_encoded_video(path, len(indices), 30)
        video_metadata[camera] = dict(**video_probe(path), **identity(path))
    record = dict(method=method, floor=int(reset["floor"]), run_id=meta["run_id"],
        source_env_id=int(meta["env_id"]), source_trajectory=str(episode["path"]),
        source_trajectory_sha256=source["sha256"], source_metadata_sha256=source_meta["sha256"],
        source_metadata=source_meta, source_physics=source,
        layout=dict(offset_x_m=reset["offset_x_m"], offset_y_m=reset["offset_y_m"]),
        reset_episode=reset, seed=meta["seed"], success=bool(info["success"]),
        termination=info["termination"], pressed_floors=info.get("pressed_floors", []),
        sim_seconds=float(info["sim_seconds"]), fps=30, frame_count=len(indices),
        duration_seconds=len(indices) / 30., frame_sim_seconds=[row["sim_time"] for row in frame_rows],
        physics_indices=indices, frame_records=frame_rows,
        global_video=str((directory / "global.mp4").relative_to(output)),
        wrist_video=str((directory / "wrist.mp4").relative_to(output)),
        initial_global_image=str((directory / "start_global.jpg").relative_to(output)),
        terminal_global_image=str((directory / "end_global.jpg").relative_to(output)),
        videos=video_metadata, restoration_checks=checks, physics_steps_during_replay=0,
        renderer_world_time_start=time_before, renderer_world_time_end=float(backend.world.current_time),
        video_timing="30 Hz CFR: t=0, every fourth measured 120 Hz tick, plus exact terminal tick; "
            "a partial terminal interval is displayed at the next CFR slot; no time interpolation",
        rendering="Measured physical states restored into one environment, camera native 640x480; "
            "no policy inference or action/controller execution during replay. Presentation RGB is newly rendered.")
    write_json(directory / "video_metadata.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    frozen = ROOT / "outputs/rtx5090_eval_comparison/input"
    parser.add_argument("--trajectories", nargs="+", type=Path, required=True)
    parser.add_argument("--run-map", type=Path, required=True, help="JSON mapping captured run UUIDs to methods")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=Path, default=frozen / "config.json")
    parser.add_argument("--snapshot", type=Path, default=frozen / "scene.usda")
    parser.add_argument("--dataset", type=Path, default=frozen / "dataset_metadata")
    parser.add_argument("--asset-bundle", type=Path, default=frozen / "assets")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--floors", nargs="+", type=int, help="Optional preselected floors to render")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error(f"Output must be new: {args.output}")
    if args.gpu < 0 or args.cpu_threads < 1:
        parser.error("GPU and thread settings must be valid")
    run_map = json.loads(args.run_map.read_text())
    episodes = discover(args.trajectories, run_map)
    if args.floors:
        episodes = [item for item in episodes if item["metadata"]["reset_episode"]["floor"] in args.floors]
        if not episodes:
            parser.error("No trajectories match the requested floors")
    # Validate all source data on CPU before opening a Kit/GPU context.
    for episode in episodes:
        load_trajectory(episode)
    args.output.mkdir(parents=True)
    native_log = args.output / "renderer.kit.log"
    for key, value in {"OMNI_KIT_ACCEPT_EULA": "YES", "PXR_WORK_THREAD_LIMIT": "8",
                       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}.items():
        os.environ.setdefault(key, value)
    tmp = ROOT / ".cache/tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmp))
    manifest = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
        status="rendering", run_map=run_map, expected_episodes=len(episodes), episodes=[],
        source_script=identity(__file__), camera_resolution=[640, 480], fps=30,
        source_physics_hz=120, gpu=args.gpu, gpu_dynamics=True,
        initialization="Isaac scene setup and one 90-tick home reset precede playback; "
            "all measured state variables are overwritten before every rendered frame.",
        physics_steps_during_replay=0,
        caveat="New presentation renders of measured rollout states; not the original inference RGB frames.")
    write_json(args.output / "render_manifest.json", manifest)
    app = backend = None
    exit_code = 1

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        from isaacsim import SimulationApp
        from collect_dataset import DATASET_ANTI_ALIASING
        app = SimulationApp(dict(headless=True, create_new_stage=False, width=640, height=480,
            active_gpu=args.gpu, physics_gpu=args.gpu, multi_gpu=False, disable_viewport_updates=True,
            limit_cpu_threads=args.cpu_threads, renderer="RayTracedLighting", anti_aliasing=DATASET_ANTI_ALIASING,
            fast_shutdown=True, extra_args=["--/app/asyncRendering=false", "--/rtx/ecoMode/enabled=false",
                f"--/log/file={native_log}", "--/log/fileFlushLevel=verbose", "--/log/fileAppend=false",
                "--/log/async=false", "--/plugins/carb.tasking.plugin/threadCount=8",
                "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8", "--/persistent/physics/numThreads=4",
                "--/validate/p2p/enabled=false", "--/validate/iommu/enabled=false"]))
        from pressb.online_rl.fast_simulation import FastIsaacVectorBackend
        backend = FastIsaacVectorBackend(app, project_root=ROOT, config=args.config.resolve(),
            snapshot=args.snapshot.resolve(), dataset=args.dataset.resolve(), output=args.output / "renderer",
            num_envs=1, max_seconds=15., smoothing_window=3, asset_bundle=args.asset_bundle.resolve(),
            gpu=args.gpu, camera_resolution=640, render_subframes=1, light_settle_captures=1,
            light_settle_subframes=2, ik_iterations=16, log_actions=False, png_workers=1,
            cpu_threads=args.cpu_threads, gpu_dynamics=True, native_log_path=native_log)
        first = episodes[0]["metadata"]
        backend.reset([first["reset_episode"]], int(first["seed"]))
        backend.recording.clear()

        def forbidden_physics_step(*args, **kwargs):
            raise RuntimeError("Physics stepping is prohibited during measured trajectory rendering")

        backend.world.step = forbidden_physics_step
        manifest["renderer_initial_world_time"] = float(backend.world.current_time)
        manifest["initialization_backend_ticks"] = int(backend._world_ticks)
        for episode in episodes:
            result = render_episode(backend, episode, args.output)
            manifest["episodes"].append(result)
            write_json(args.output / "render_manifest.json", manifest)
            print(json.dumps(dict(event="episode_rendered", method=result["method"], floor=result["floor"],
                frames=result["frame_count"], success=result["success"], completed=len(manifest["episodes"]),
                total=len(episodes))), flush=True)
        manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
            renderer_final_world_time=float(backend.world.current_time))
        exit_code = 0
    except (Exception, KeyboardInterrupt) as error:
        traceback.print_exc()
        manifest.update(status="failed", error=repr(error))
    finally:
        write_json(args.output / "render_manifest.json", manifest)
        try:
            if backend is not None:
                backend.close()
        finally:
            if app is not None:
                try:
                    app.app.post_quit(exit_code)
                finally:
                    app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
