#!/usr/bin/env python3
"""Render recorded PAP physical state at native resolution and real-time speed.

The generated RGB is an offline visualization of measured joints/object poses,
not policy input or a fresh execution. Physics stepping is forbidden in replay.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def audit_clock(data, physics_dt):
    """Isaac's World clock accumulates the float32 PhysX dt, not Python dt."""
    ticks = data["episode_physics_steps"]
    elapsed = data["world_time"]-data["world_time"][0]
    runtime_dt = float(np.float32(physics_dt))
    runtime_error = float(np.max(np.abs(elapsed-ticks*runtime_dt)))
    declared_error = float(np.max(np.abs(elapsed-ticks*physics_dt)))
    tolerance = 1e-8
    if runtime_error > tolerance:
        raise ValueError("PAP recorded world clock disagrees with ticks times PhysX float32 dt")
    return dict(declared_physics_dt_seconds=float(physics_dt),
        physx_float32_dt_seconds=runtime_dt, max_world_vs_runtime_error_seconds=runtime_error,
        runtime_tolerance_seconds=tolerance, max_world_vs_declared_error_seconds=declared_error,
        playback_clock="integer physics ticks multiplied by declared physics_dt; no accumulated wall-clock rounding")


def replay_config(manifest, output, gpu):
    """Reconstruct raw objects before the production loader normalizes them.

    load_config preserves the original object dictionary inside ``geometry``;
    feeding its normalized output back into it would nest another geometry and
    silently change the task identity. Require exact normalized-object equality.
    """
    from pressb.rl_nodes.pap_vector import load_config
    config = copy.deepcopy(manifest["config"])
    original_objects = copy.deepcopy(config["objects"])
    raw_objects = []
    for obj in original_objects:
        raw = copy.deepcopy(obj["geometry"])
        raw.pop("size_xyz", None)
        raw.pop("grasp_width_m", None)
        raw_objects.append(raw)
    config.update(num_envs=1, gpu_ids=[gpu], gpu=gpu, workers=[{"gpu": gpu}],
        render_resolution=[640, 480], log_dir=str(output/"renderer_logs"),
        scene_overrides=copy.deepcopy(config["scene"]), objects=raw_objects)
    config.pop("trajectory_recording", None)
    normalized = load_config(config)
    if normalized["objects"] != original_objects:
        raise ValueError("Cannot reconstruct the exact original PAP object input for rendering")
    return config


def load_episode(path):
    path = path.resolve()
    meta = json.loads(path.read_text())
    if meta.get("format") != "pap_measured_state" or meta.get("schema_version") != 1 or not meta.get("complete"):
        raise ValueError(f"A complete PAP measured-state recording is required: {path}")
    source = path.with_suffix(".npz")
    manifest_path = path.parent/"manifest.json"
    for actual_path, expected in ((source, meta["trajectory"]), (manifest_path, meta["manifest"])):
        if identity(actual_path)["sha256"] != expected["sha256"]:
            raise ValueError(f"Recorded input checksum mismatch: {actual_path}")
    manifest = json.loads(manifest_path.read_text())
    with np.load(source, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    ticks = data["episode_physics_steps"]
    dt, fps = float(manifest["physics_dt"]), float(manifest["control_hz"])
    stride = round(1/dt/fps)
    if not math.isclose(fps, 30.) or not math.isclose(stride*dt*fps, 1.):
        raise ValueError("Renderer currently requires the PAP 30 Hz control contract")
    if (ticks.ndim != 1 or ticks[0] != 0 or not np.all(np.diff(ticks) > 0)
            or not np.all(ticks[:-1] % stride == 0) or np.any(np.diff(ticks) > stride)
            or ticks[-1] != meta["final_physics_tick"] or ticks[-1] != meta["info"]["episode_physics_steps"]):
        raise ValueError("PAP sample timing or exact terminal tick is invalid")
    if any(len(value) != len(ticks) or not np.isfinite(value).all() for value in data.values()):
        raise ValueError("PAP recorded state shape/finiteness check failed")
    audit_clock(data, dt)
    duration = float(ticks[-1]*dt)
    if not math.isclose(duration, meta["duration_seconds"], abs_tol=1e-9):
        raise ValueError("PAP recorded duration disagrees with physics ticks")
    times = ticks*dt
    output_times = np.arange(math.ceil(duration*fps-1e-9)+1)/fps
    samples = np.searchsorted(times, output_times+1e-9, side="right")-1
    samples = np.clip(samples, 0, len(times)-1)
    if samples[-1] != len(times)-1:
        raise ValueError("Output must include the exact terminal state")
    return meta, manifest, data, samples


def render_episode(backend, path, output, *, render_captures=None, render_subframes=None):
    from PIL import Image
    from collect_dataset import VideoPipe, validate_encoded_video
    meta, manifest, data, samples = load_episode(path)
    directory = output/path.stem
    directory.mkdir(exist_ok=False)
    recorded_names = manifest["joint_names"]
    actual_names = list(backend.arms.view.shared_metatype.dof_names)
    if len(set(recorded_names)) != len(recorded_names) or set(actual_names) != set(recorded_names):
        raise ValueError("Recorded and rendered robot DOF names differ")
    order = [recorded_names.index(name) for name in actual_names]
    backend._reset([0], {0: copy.deepcopy(meta["reset"])}, capture_images=False)
    world_time = float(backend.world.current_time)
    offset = np.asarray(meta["source_env_offset"])
    arm_rows, object_rows = backend.arm_rows[[0]], backend.object_rows[[0]]
    captures = render_captures or backend.config["observation_render_frames"]
    subframes = render_subframes or backend.config["render_subframes"]
    pipes, frames = {}, []
    errors = dict(joint_position=0., joint_velocity=0., object_position_m=0., object_quaternion=0., object_velocity=0.)
    try:
        pipes = {name: VideoPipe(directory/f"{name}.mp4", 30) for name in ("up", "wrist")}
        for frame, sample in enumerate(samples):
            q = data["joints"][sample, order][None]
            qv = data["joint_velocity"][sample, order][None]
            p = (data["object_position_world"][sample]-offset+backend.offsets[0])[None]
            quat = data["object_quaternion_wxyz"][sample][None]
            v = data["object_velocity"][sample][None]
            backend.arms.set_joint_positions(q, arm_rows)
            backend.arms.set_joint_velocities(qv, arm_rows)
            backend.arms.set_joint_position_targets(q, arm_rows)
            backend.bodies.set_world_poses(p, quat, object_rows)
            backend.bodies.set_velocities(v, object_rows)
            backend._begin_render_boundary()
            # Presentation-only quality override, after validating the source
            # task identity. Restore the configuration even on failure; these
            # extra zero-time samples never enter policy observations.
            original_subframes = backend.config["render_subframes"]
            try:
                backend.config["render_subframes"] = subframes
                for _ in range(captures-1):
                    backend._capture(return_tiles=False)
                tiles = backend._capture()
            finally:
                backend.config["render_subframes"] = original_subframes
            got_p, got_quat = backend.bodies.get_world_poses()
            current = dict(joint_position=float(np.max(np.abs(backend.arms.get_joint_positions()[arm_rows]-q))),
                joint_velocity=float(np.max(np.abs(backend.arms.get_joint_velocities()[arm_rows]-qv))),
                object_position_m=float(np.max(np.abs(got_p[object_rows]-p))),
                object_quaternion=float(np.max(np.abs(got_quat[object_rows]-quat))),
                object_velocity=float(np.max(np.abs(backend.bodies.get_velocities()[object_rows]-v))))
            if any(value > 2e-6 for value in current.values()) or float(backend.world.current_time) != world_time:
                raise RuntimeError(f"Measured state/clock changed during rendering: {current}")
            errors = {key: max(errors[key], current[key]) for key in errors}
            for camera, tile in zip(("up", "wrist"), tiles):
                pipes[camera].add(tile)
                if frame in (0, len(samples)-1):
                    Image.fromarray(tile).save(directory/f"{'start' if frame == 0 else 'end'}_{camera}.png")
            frames.append(dict(frame=frame, video_seconds=frame/30., source_sample=int(sample),
                physics_tick=int(data["episode_physics_steps"][sample]),
                sim_seconds=float(data["episode_physics_steps"][sample]*manifest["physics_dt"])))
            if frame % 150 == 0:
                print(json.dumps(dict(event="render_progress", episode=path.stem, frame=frame, total=len(samples))), flush=True)
    finally:
        for pipe in pipes.values():
            pipe.close()
    for camera in pipes:
        validate_encoded_video(directory/f"{camera}.mp4", len(samples), 30)
    result = dict(status="complete", source_metadata=identity(path), source_trajectory=meta["trajectory"],
        env_id=meta["env_id"], reset=meta["reset"], success=bool(meta["info"]["success"]),
        termination_reason=meta["info"]["termination_reason"], sim_seconds=meta["duration_seconds"],
        fps=30, frame_count=len(samples), videos={camera: identity(directory/f"{camera}.mp4") for camera in pipes},
        frame_records=frames, restoration_errors=errors, physics_steps_during_replay=0,
        clock_audit=audit_clock(data, manifest["physics_dt"]),
        presentation_rendering=dict(captures_per_frame=captures, subframes_per_capture=subframes,
            reset_dlss_history_each_frame=True,
            recorded_captures_per_observation=backend.config["observation_render_frames"],
            recorded_subframes_per_capture=backend.config["render_subframes"],
            scope="Offline presentation only; source task and camera identity validated before overrides"),
        world_time_before=world_time, world_time_after=float(backend.world.current_time),
        presentation="Native 640x480 newly rendered measured-state replay; actual joints and free object pose. "
            "No policy execution, physics stepping, synthetic grasp, motion interpolation or time compression. "
            "Exact terminal tick appears at the next 30 Hz display slot (delay < 1/30 s).")
    write_json(directory/"render_manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, nargs="+", required=True, help="Recorded episode JSON files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--render-captures", type=int, help="Offline zero-time captures per video frame; default matches recorded observations")
    parser.add_argument("--render-subframes", type=int, default=8,
                        help="Offline subframes per capture (default: 8 to reduce RTX shimmer; 1 reproduces fast rollout rendering)")
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("gpu must be nonnegative")
    if any(value is not None and value < 1 for value in (args.render_captures, args.render_subframes)):
        parser.error("render captures and subframes must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    entries = [load_episode(path) for path in args.episodes]
    manifest = entries[0][1]
    if any(entry[1] != manifest for entry in entries):
        raise ValueError("All episodes must share the same recording manifest")
    config = replay_config(manifest, args.output, args.gpu)
    from pressb.rl_nodes.pap_vector import PAPVectorBackend
    backend = PAPVectorBackend(config)
    try:
        if backend.identity["task_config_sha256"] != manifest["task_identity"]["task_config_sha256"]:
            raise ValueError("Replay task semantics differ from the recorded rollout")
        def forbidden_step(*args, **kwargs):
            raise RuntimeError("Physics stepping is prohibited during measured-state replay")
        backend.world.step = forbidden_step
        results = [render_episode(backend, path.resolve(), args.output,
            render_captures=args.render_captures, render_subframes=args.render_subframes) for path in args.episodes]
        write_json(args.output/"render_manifest.json", dict(status="complete", episodes=results,
            source_manifest=identity(args.episodes[0].parent/"manifest.json"), renderer_identity=backend.identity))
        print(json.dumps(dict(event="render_complete", output=str(args.output), episodes=len(results))), flush=True)
    except BaseException as error:
        # SimulationApp.close() can exit before Python propagates an exception.
        # Persist and flush the failure before closing the Isaac application.
        write_json(args.output/"error.json", dict(status="failed", exception=type(error).__name__,
            message=str(error), traceback=traceback.format_exc()))
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        backend.close()


if __name__ == "__main__":
    main()
