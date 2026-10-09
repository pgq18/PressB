#!/usr/bin/env python3
"""Collect independently simulated, synchronized dual-camera Piper demonstrations.

Each atomically committed raw episode can be converted again without rerunning
physics. No Kit, video writer, or dataset root is shared between processes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RAW_SCHEMA_VERSION = 11
LIGHTING_POLICY = {
    "global_dome": "deactivate_duplicate_dome_prims",
    "active_dome_count": 1,
    "shared_dome_environment": 0,
    "local_rect_lights_per_environment": 2,
}
DATASET_ANTI_ALIASING = 2  # FXAA: no temporal anti-aliasing history across samples.
LIGHT_SETTLE_SUBFRAMES = 16
# Kit 107 needs additional captures to drain visible edge-light history.
# All settling captures keep physics frozen and are recorded in the manifest.
LIGHT_SETTLE_CAPTURES = 16
RENDER_EXPOSURE_CONTROLS = {
    "/rtx/post/histogram/enabled": False,
    "/rtx/post/tonemap/filmIso": 100.,
    "/rtx/post/tonemap/exposureTime": .02,
    "/rtx/post/tonemap/fNumber": 5.,
    "/rtx/post/tonemap/op": 6,
}
# NRD 3.2.3 permits zero history. Keep spatial denoising, but do not mix a
# previous illuminated frame into the current unlit observation.
RENDER_HISTORY_CONTROLS = {
    "/rtx/indirectDiffuse/denoiser/temporal/enabled": False,
    "/rtx/directLighting/sampledLighting/ris/enableSpatioTemporalRis": False,
    **{f"/rtx/lightspeed/NRD_Reblur{channel}/{key}": 0
       for channel in ("Diffuse", "Specular")
       for key in ("maxAccumulatedFrameNum", "maxFastAccumulatedFrameNum")},
    **{f"/rtx/lightspeed/NRD_ReLAX/{key}": 0 for key in
       ("diffuseHistoryFrames", "diffuseFastHistoryFrames", "specularHistoryFrames", "specularFastHistoryFrames")},
}
SOURCE_NAMES = ("frames.npz", "physics.npz", "wrist.mp4", "global.mp4")
START_SPEED_TOLERANCE_RAD_S = .02


def capture_stride_for(fps, physics_dt=1 / 120):
    """Keep every RGB/state sample on an exact 120 Hz physics boundary."""
    if (type(fps) is not int or fps <= 0 or 120 % fps
            or not math.isfinite(physics_dt) or abs(physics_dt - 1 / 120) > 1e-12):
        raise ValueError("Require integer FPS dividing 120 and physics_dt=1/120 seconds")
    return 120 // fps


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def file_identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def prepare_collection_manifest(args, cfg):
    """Create once, or verify immutable collection identity before starting Kit.

    Environment/worker counts and GPU assignments deliberately do not identify
    the collection: every episode records its own environment world offset.
    The lock also prevents concurrent workers from replacing the first manifest.
    """
    stride = capture_stride_for(args.fps, float(cfg["physics_dt"]))
    camera_metadata = {}
    camera_sources = {}
    for view in ("wrist", "global"):
        path = args.snapshot.parent / f"{view}_camera/intrinsics.json"
        camera_metadata[view] = json.loads(path.read_text())
        if (camera_metadata[view].get("width"), camera_metadata[view].get("height")) != (640, 480):
            raise ValueError(f"Expected 640x480 snapshot camera calibration: {path}")
        camera_sources[view] = {"path": str(path), **file_identity(path)}
    identity = {
        "raw_schema_version": RAW_SCHEMA_VERSION, "config": cfg,
        "scene_sha256": file_identity(args.snapshot)["sha256"], "seed": args.seed,
        "collection_code_sha256": {name: file_identity(ROOT / name)["sha256"] for name in (
            "scripts/collect_dataset.py", "src/pressb/dataset_scene.py", "src/pressb/dataset_planning.py",
            "src/pressb/scene.py", "src/pressb/planning.py", "src/pressb/panel_randomization.py",
            "src/pressb/panel_metadata.py")},
        "lighting_policy": LIGHTING_POLICY,
        "fps": args.fps, "physics_hz": 120, "capture_stride": stride,
        "action_horizon_s": 1 / args.fps,
        "capture_method": "replicator_step_zero_dt_reset_accumulation", "render_subframes_per_capture": 2,
        "light_change_settle_subframes": LIGHT_SETTLE_SUBFRAMES,
        "light_change_settle_captures": LIGHT_SETTLE_CAPTURES,
        "renderer_history_controls": RENDER_HISTORY_CONTROLS,
        "renderer_exposure_controls": RENDER_EXPOSURE_CONTROLS,
        "anti_aliasing": DATASET_ANTI_ALIASING,
        "pose_frame": "base_link", "pose_link": "gripper_tcp", "quaternion_order": "wxyz",
        "tcp_offset_link6_m": [0, 0, .1358], "press_tip_offset_link6_m": [0, 0, .24],
        "gripper_width_m": .008,
        "action_semantics": f"absolute planned TCP target at min(current physics index + {stride}, final plan index)",
        "camera_calibration_sha256": {view: camera_sources[view]["sha256"] for view in camera_sources},
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                             allow_nan=False).encode()).hexdigest()
    manifest = {**identity, "collection_fingerprint": fingerprint, "identity": identity,
        "pose_names": ["x_m", "y_m", "z_m", "qw", "qx", "qy", "qz", "gripper_width_m"],
        "task_texts": [f"Press {f} floor." for f in range(24, 36)],
        "sample_time_origin": "first captured post-physics state",
        "first_sample_physics_time_s": float(cfg["physics_dt"]),
        "source_snapshot": str(args.snapshot), "source_calibrations": camera_sources,
        "created_at": datetime.now(timezone.utc).isoformat(), **camera_metadata}
    manifest_path = args.output / "collection_metadata.json"
    with (args.output / ".collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if (existing.get("collection_fingerprint") != fingerprint
                    or existing.get("identity") != identity
                    or any(existing.get(key) != value for key, value in identity.items())
                    or any(existing.get(view) != camera_metadata[view] for view in camera_metadata)):
                raise ValueError(f"Collection identity mismatch; existing manifest is unchanged: {manifest_path}")
            return existing
        if any(args.output.glob("episode_*")):
            raise ValueError("Committed episodes exist without a collection manifest; use a new output directory")
        write_json(manifest_path, manifest)
    return manifest


def validate_source_arrays(directory, metadata):
    """Verify durable source shape, finiteness and physics-to-frame alignment."""
    import numpy as np
    fps = metadata["fps"]
    stride = capture_stride_for(fps)
    if (metadata.get("physics_hz", 120) != 120 or metadata.get("capture_stride", stride) != stride
            or not math.isclose(metadata.get("action_horizon_s", 1 / fps), 1 / fps, rel_tol=0, abs_tol=1e-12)):
        raise ValueError(f"Inconsistent episode capture timing: {directory}")
    if metadata.get("raw_schema_version", 0) >= 10 and any(
            key not in metadata for key in ("physics_hz", "capture_stride", "action_horizon_s")):
        raise ValueError(f"Missing episode capture timing: {directory}")
    count, steps = metadata["num_frames"], metadata["physics_steps"]
    if count < 2 or steps != (count - 1) * stride + 1:
        raise ValueError(f"Invalid padded episode/sample lengths: {directory}")
    with np.load(directory / "frames.npz", allow_pickle=False) as frames, \
            np.load(directory / "physics.npz", allow_pickle=False) as physics:
        shapes = {"state": (count, 8), "action": (count, 8), "sim_time": (count,),
                  "q_actual": (count, 6), "q_target": (count, 6), "physics_index": (count,),
                  "phase": (count,), "lights": (count, 12)}
        physics_shapes = {"q_actual": (steps, 6), "q_command": (steps, 6),
                          "gripper_actual": (steps, 2), "button_travel": (steps, 12),
                          "contact_force": (steps, 12), "lights": (steps, 12)}
        for source, expected in ((frames, shapes), (physics, physics_shapes)):
            if set(source.files) != set(expected):
                raise ValueError(f"Incomplete source array schema: {directory}")
            for key, shape in expected.items():
                values = source[key]
                if values.shape != shape or (key != "phase" and not np.isfinite(values).all()):
                    raise ValueError(f"Invalid source array {key}: {directory}")
        indices = np.arange(count) * stride
        plan_samples = metadata["plan_physics_steps"]
        if not 1 <= plan_samples <= steps or steps - plan_samples >= stride:
            raise ValueError(f"Invalid terminal padding length: {directory}")
        future = np.minimum(indices + stride, plan_samples - 1)
        # sample_state_action promotes the measured PhysX float32 joints to
        # float64 before subtraction; preserve that operation order here.
        fingers = physics["gripper_actual"][indices].astype(np.float64)
        if (not np.array_equal(frames["physics_index"], indices)
                or not np.allclose(frames["sim_time"], np.arange(count) / fps, rtol=0, atol=1e-10)
                or not np.array_equal(frames["q_actual"], physics["q_actual"][indices])
                or not np.array_equal(frames["q_target"], physics["q_command"][future])
                or not np.array_equal(frames["lights"], physics["lights"][indices])
                or not np.allclose(frames["state"][:, 7], fingers[:, 0] - fingers[:, 1], rtol=0, atol=1e-12)
                or not np.allclose(frames["action"][:, 7], .008, rtol=0, atol=1e-12)
                or any(not np.allclose(np.linalg.norm(frames[key][:, 3:7], axis=1), 1., atol=1e-6)
                       for key in ("state", "action"))):
            raise ValueError(f"Physics/frame/action alignment mismatch: {directory}")


def validate_committed_episode(directory, episode_id, args, manifest):
    from pressb.panel_metadata import validate_episode_panel_metadata
    metadata = json.loads((directory / "metadata.json").read_text())
    validate_episode_panel_metadata(manifest, metadata)
    floor = 24 + episode_id % 12
    expected = {"raw_schema_version": manifest["raw_schema_version"], "success": True,
                "collection_fingerprint": manifest["collection_fingerprint"], "episode_id": episode_id,
                "floor": floor, "seed": args.seed + episode_id, "task": f"Press {floor} floor.",
                **{key: manifest[key] for key in ("fps", "physics_hz", "capture_stride", "action_horizon_s")}}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Invalid committed episode identity: {directory}")
    sources = metadata.get("source_files", {})
    if set(sources) != set(SOURCE_NAMES):
        raise ValueError(f"Missing committed source checksums: {directory}")
    for name in SOURCE_NAMES:
        path = directory / name
        if not path.is_file() or file_identity(path) != sources[name]:
            raise ValueError(f"Missing or changed committed source: {path}")
    validate_source_arrays(directory, metadata)


def pending_episode_ids(args, manifest):
    todo = []
    for episode_id in range(12 * args.episodes_per_task):
        if episode_id % args.workers != args.worker_index:
            continue
        directory = args.output / f"episode_{episode_id:06d}"
        if directory.exists():
            validate_committed_episode(directory, episode_id, args, manifest)
        else:
            todo.append(episode_id)
    # Group targets with similar motion duration so shorter demonstrations do
    # not spend each batch waiting for the highest button. IDs and seeds remain
    # fixed and independent of this execution order.
    todo.sort(key=lambda episode_id: (episode_id % 12, episode_id // 12))
    return todo[:args.max_episodes] if args.max_episodes else todo


def validate_encoded_video(path, expected_frames, fps):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                             "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
                             "-of", "json", str(path)], check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError(f"Invalid encoded video stream: {path}")
    stream = streams[0]
    numerator, denominator = (int(value) for value in stream["avg_frame_rate"].split("/"))
    if (stream["width"] != 640 or stream["height"] != 480
            or int(stream["nb_read_frames"]) != expected_frames or denominator <= 0
            or numerator != fps * denominator):
        raise ValueError(f"Encoded video dimensions, frame count or timing mismatch: {path}")


class VideoPipe:
    def __init__(self, path, fps):
        self.log = path.with_suffix(".log").open("wb")
        self.process = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", "640x480", "-r", str(fps), "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-threads", "2", "-preset", "veryfast",
            "-crf", "16", "-g", str(fps), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(path)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        self.frames = 0

    def add(self, rgb):
        self.process.stdin.write(rgb.tobytes())
        self.frames += 1

    def close(self):
        self.process.stdin.close()
        result = self.process.wait(timeout=60)
        self.log.close()
        if result:
            raise RuntimeError(f"FFmpeg exited {result}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/piper_elevator_raw_panel_randomized_30hz")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dataset_panel_randomized.json")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "outputs/edge_feedback/scene.usda")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30, help="Synchronized RGB/state/action FPS; must divide 120")
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reset-renderer-accumulation", action="store_true", default=True,
                        help="Compatibility flag: renderer history is always reset at lamp transitions")
    args = parser.parse_args()
    try:
        capture_stride_for(args.fps)
    except ValueError as exc:
        parser.error(str(exc))
    if min(args.num_envs, args.episodes_per_task, args.workers) < 1 or not 0 <= args.worker_index < args.workers:
        parser.error("Invalid environment/task/worker counts")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    worker_lock = (args.output / f".worker_{args.worker_index:02d}.lock").open("a")
    try:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        worker_lock.close()
        raise RuntimeError(f"Worker {args.worker_index} is already collecting into {args.output}") from exc
    args.snapshot, args.config = args.snapshot.resolve(), args.config.resolve()
    cfg = json.loads(args.config.read_text())
    if cfg.get("panel_randomization", {}).get("sampling_mode") == "stratified_grid":
        from pressb.panel_randomization import panel_randomization_settings
        settings = panel_randomization_settings(cfg)
        if math.prod(settings["grid_shape"]) != args.episodes_per_task:
            parser.error("Stratified collection requires one episode per grid cell for each floor")
        if settings["grid_seed"] != args.seed:
            parser.error("Stratified grid_seed must match the collection --seed")
    # The dataset CLI owns capture timing. Save the effective configuration,
    # rather than retaining a misleading single-scene preview stride.
    stride = capture_stride_for(args.fps, float(cfg["physics_dt"]))
    for key in ("render_stride", "wrist_camera_capture_stride", "global_camera_capture_stride"):
        cfg[key] = stride
    status = {"status": "starting", "started_at": datetime.now(timezone.utc).isoformat(),
              "num_envs": args.num_envs, "gpu": args.gpu, "episodes_per_task": args.episodes_per_task,
              "fps": args.fps}
    status_path = args.output / f"worker_{args.worker_index:02d}_status.json"
    write_json(status_path, status)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    app = None
    try:
        manifest = prepare_collection_manifest(args, cfg)
        todo = pending_episode_ids(args, manifest)
        if not todo:
            status.update(status="complete", scheduled=0, completed_this_run=0,
                          finished_at=datetime.now(timezone.utc).isoformat())
            return
        from isaacsim import SimulationApp
        # World creates the stage below. A headless collector does not need a
        # viewport handle; waiting for one can stall concurrent Kit instances.
        app = SimulationApp({"headless": True, "create_new_stage": False, "width": 640, "height": 480,
            "active_gpu": args.gpu, "physics_gpu": args.gpu, "multi_gpu": False,
            "renderer": "RayTracedLighting", "anti_aliasing": DATASET_ANTI_ALIASING, "fast_shutdown": True,
            "extra_args": ["--/app/asyncRendering=false", "--/rtx/ecoMode/enabled=false",
                           "--/plugins/carb.tasking.plugin/threadCount=8",
                           "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8",
                           "--/persistent/physics/numThreads=4",
                           "--/validate/p2p/enabled=false", "--/validate/iommu/enabled=false"]})
        run(app, args, cfg, status, status_path, manifest, todo)
        status.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        write_json(status_path, status)
        if app:
            app.app.post_quit(0 if status["status"] == "complete" else 1)
            app.close()
        worker_lock.close()


def run(app, args, cfg, status, status_path, manifest, todo):
    import numpy as np
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom, UsdLux, PhysicsSchemaTools
    import omni.physx
    import omni.replicator.core as rep
    from isaacsim.core.api import World
    from isaacsim.core.utils.types import ArticulationAction
    from pressb.dataset_scene import create_envs, build_tiled_rgb, set_panel_offset, validate_panel_layout
    from pressb.dataset_planning import make_episode_plan, sample_state_action
    from pressb.kinematics import PiperKinematics
    from pressb.scene import set_button_light

    kin = PiperKinematics(ROOT / cfg["robot_urdf"], tip_offset=cfg["tip_offset"])
    fps, dt = args.fps, float(cfg["physics_dt"])
    stride = capture_stride_for(fps, dt)
    world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=1 / fps,
                  backend="numpy", device="cpu")
    world.get_physics_context().set_physx_update_transformations_settings(
        update_to_usd=True, update_velocities_to_usd=True)
    envs = create_envs(world, args.snapshot, args.num_envs, cfg=cfg)
    domes = [{"path": str(prim.GetPath()), "active": prim.IsActive(),
              "intensity": UsdLux.DomeLight(prim).GetIntensityAttr().Get()}
             for prim in world.stage.TraverseAll() if prim.IsA(UsdLux.DomeLight)]
    local_lights = [[str(prim.GetPath()) for prim in Usd.PrimRange(world.stage.GetPrimAtPath(env.prefix))
                     if prim.IsA(UsdLux.RectLight)] for env in envs]
    if (len(domes) != args.num_envs or sum(light["active"] for light in domes) != 1
            or not domes[0]["active"] or any(len(lights) != 2 for lights in local_lights)):
        raise RuntimeError(f"Unexpected collection lighting topology: {domes}, {local_lights}")
    write_json(args.output / f"worker_{args.worker_index:02d}_lighting.json", {
        "collection_fingerprint": manifest["collection_fingerprint"],
        "lighting_policy": LIGHTING_POLICY, "domes": domes, "local_rect_lights": local_lights})
    camera_paths = [p for env in envs for p in (env.wrist_camera_path, env.global_camera_path)]
    product, annotator, split_rgb = build_tiled_rgb(camera_paths)
    rep.orchestrator.set_capture_on_play(False)
    world.reset()
    controls = []
    home = np.asarray(cfg["home_q"])
    gripper = np.asarray(cfg["gripper_joint_positions_m"])
    base = np.array([cfg["robot_base_x"], cfg["robot_base_y"], cfg["table_height"]])
    for env in envs:
        robot = env.robot
        arm_i = np.array([robot.get_dof_index(name) for name in kin.joint_names])
        finger_i = np.array([robot.get_dof_index(name) for name in ("joint7", "joint8")])
        command = robot.get_joint_positions().copy()
        command[arm_i], command[finger_i] = home, gripper
        robot.set_joint_positions(command)
        robot.set_joint_velocities(np.zeros_like(command))
        controller = robot.get_articulation_controller()
        controller.set_gains(kps=np.full(len(command), 12000.), kds=np.full(len(command), 600.))
        controller.apply_action(ArticulationAction(joint_positions=command))
        controls.append((arm_i, finger_i, command, controller))
    forces = np.zeros((len(envs), 12))
    grasp_forces = np.zeros(len(envs))
    unexpected = [[] for _ in envs]
    contact_step = [-1]
    body_map = {info["body_path"]: (env.env_id, f) for env in envs for f, info in env.button_info.items()}
    actor_env = {f"{env.prefix}/Piper/": env.env_id for env in envs}
    tool_paths = {p for env in envs for p in env.tool_colliders}

    def on_contact(headers, data):
        for header in headers:
            if not header.num_contact_data:
                continue
            actors = [str(PhysicsSchemaTools.intToSdfPath(x)) for x in (header.actor0, header.actor1)]
            colliders = [str(PhysicsSchemaTools.intToSdfPath(x)) for x in (header.collider0, header.collider1)]
            pair = next((body_map[a] for a in actors if a in body_map), None)
            force = sum(float(np.linalg.norm(data[k].impulse)) / dt for k in
                        range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data))
            if pair is not None and any(c in tool_paths for c in colliders):
                forces[pair[0], pair[1] - 24] += force
                continue
            for prefix, eid in actor_env.items():
                robot_actors = [a for a in actors if a.startswith(prefix)]
                if not robot_actors or force <= .1:
                    continue
                # The fingers intentionally clamp the 8 mm shaft. Keep this
                # physical contact enabled and measured; it is not an arm-arm
                # collision. Match only the two fingertip/shaft collider pairs.
                shaft = prefix + "link6/PressStylus"
                pads = {prefix + "link7/collisions", prefix + "link8/collisions"}
                if shaft in colliders and any(c in pads for c in colliders):
                    grasp_forces[eid] = max(grasp_forces[eid], force)
                    continue
                reason = None
                if pair is not None:
                    reason = "non_stylus_button_contact"
                elif len(robot_actors) == 2 and actors[0] != actors[1]:
                    reason = "robot_self_contact"
                elif len(robot_actors) == 1 and robot_actors[0].rsplit("/", 1)[-1] not in ("dummy_link", "base_link", "link1"):
                    reason = "robot_environment_contact"
                if reason and len(unexpected[eid]) < 100:
                    unexpected[eid].append(dict(step=contact_step[0], actors=actors, colliders=colliders,
                                                force_n=force, reason=reason))

    subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    for i in range(90):
        forces.fill(0.)
        world.step(render=False)
        if i % 4 == 0:
            world.render()
    for _ in range(8):
        world.render()
    import carb
    renderer_settings = carb.settings.get_settings()
    capture_settings = {**RENDER_HISTORY_CONTROLS, **RENDER_EXPOSURE_CONTROLS}
    previous_settings = {key: renderer_settings.get(key) for key in capture_settings}
    for key, value in capture_settings.items():
        renderer_settings.set(key, value)
    write_json(args.output / f"render_settings_worker_{args.worker_index:02d}.json",
               {"before": previous_settings, "applied": capture_settings,
                "actual": {key: renderer_settings.get(key) for key in capture_settings},
                "nrd_method": renderer_settings.get("/rtx/lightspeed/NRD/method"),
                "indirect_diffuse_denoiser_method": renderer_settings.get("/rtx/indirectDiffuse/denoiser/method")})
    # Concurrent RTX startup can finish before the first host RGB allocation.
    # Retry only this initialization capture, always at the same physical time.
    startup_time = world.current_time
    for attempt in range(12):
        rep.orchestrator.step(delta_time=0., pause_timeline=False, rt_subframes=2)
        if abs(world.current_time - startup_time) > 1e-10:
            raise RuntimeError("Camera initialization advanced physics time")
        try:
            tiles = split_rgb(annotator.get_data())
            break
        except ValueError:
            if attempt == 11:
                raise
            print(f"RGB initialization retry {attempt + 1}/12", flush=True)
    for eid, errors in enumerate(unexpected):
        if errors:
            raise RuntimeError(f"Environment {eid} warmup collisions: {errors[:2]}")
    for i, rgb in enumerate(tiles):
        if rgb.shape != (480, 640, 3) or rgb.std() < 2:
            raise RuntimeError(f"Invalid RGB camera {i}: {rgb.shape}")
        Image.fromarray(rgb).save(args.output / f"preview_worker{args.worker_index:02d}_env{i//2}_{'wrist' if i%2==0 else 'global'}.jpg")
    start_wall = time.monotonic()
    completed = 0
    status.update(status="collecting", scheduled=len(todo), completed_this_run=0)
    write_json(status_path, status)
    for batch_start in range(0, len(todo), len(envs)):
        batch = []
        for eid, episode_id in enumerate(todo[batch_start:batch_start + len(envs)]):
            floor, seed = 24 + episode_id % 12, args.seed + episode_id
            plan = make_episode_plan(kin, cfg, floor, seed, episode_index=episode_id // 12)
            env = envs[eid]
            arm_i, _, _, _ = controls[eid]
            if np.max(np.abs(env.robot.get_joint_positions()[arm_i] - home)) >= cfg["home_tolerance_rad"]:
                raise RuntimeError("Panel randomization requires the robot at folded home")
            set_panel_offset(world, env, plan.metadata["panel_offset_y_m"],
                             offset_x_m=plan.metadata["panel_offset_x_m"])
            directory = args.output / f".episode_{episode_id:06d}.inprogress"
            if directory.exists():
                directory.rename(args.output / f".episode_{episode_id:06d}.abandoned_{time.time_ns()}")
            directory.mkdir()
            batch.append(dict(id=episode_id, floor=floor, seed=seed, plan=plan, directory=directory,
                videos=[VideoPipe(directory / f"{view}.mp4", fps) for view in ("wrist", "global")],
                frames=[], physics=[], events=[], lit=set(), max_error=0., max_fk_error=0., max_gripper_error=0.,
                max_grasp_force=0., max_button_lateral_error=0.,
                length=((len(plan.q) - 1 + stride - 1) // stride) * stride + 1))
        unexpected = [[] for _ in envs]
        # Update the static housing and each spring/body at an episode boundary.
        # The arm holds home while PhysX consumes the changed joint anchors.
        # These reset ticks and discarded renders are outside all episode data.
        contact_step[0] = -1
        for _ in range(24):
            forces.fill(0.)
            world.step(render=False)
        for eid, item in enumerate(batch):
            if unexpected[eid]:
                raise RuntimeError(f"Panel reset caused contact: {unexpected[eid][:2]}")
            item["panel_layout"] = validate_panel_layout(world, envs[eid])
        import omni.usd
        omni.usd.get_context().reset_renderer_accumulation()
        reset_time = world.current_time
        for _ in range(LIGHT_SETTLE_CAPTURES):
            rep.orchestrator.step(delta_time=0., pause_timeline=False, rt_subframes=2)
        if abs(world.current_time - reset_time) > 1e-10:
            raise RuntimeError("Panel reset rendering unexpectedly advanced physics")
        unexpected = [[] for _ in envs]
        nsteps = max(item["length"] for item in batch)
        light_changed_since_capture = False
        print(f"BATCH start={batch_start} ids={[b['id'] for b in batch]} steps={nsteps}", flush=True)
        for step in range(nsteps):
            if not app.is_running():
                raise RuntimeError("Simulator closed before collection finished")
            for eid, item in enumerate(batch):
                arm_i, finger_i, command, controller = controls[eid]
                command[arm_i] = item["plan"].q[min(step, len(item["plan"].q) - 1)]
                controller.apply_action(ArticulationAction(joint_positions=command))
            forces.fill(0.)
            grasp_forces.fill(0.)
            contact_step[0] = step
            world.step(render=False)
            captures = []
            for eid, item in enumerate(batch):
                if step >= item["length"]:
                    continue
                env, plan = envs[eid], item["plan"]
                index = min(step, len(plan.q) - 1)
                arm_i, finger_i, command, controller = controls[eid]
                measured = env.robot.get_joint_positions()
                actual, fingers = measured[arm_i], measured[finger_i]
                if not np.isfinite(np.r_[actual, fingers, forces[eid], grasp_forces[eid]]).all():
                    raise RuntimeError(f"Nonfinite physics state in episode {item['id']} step {step}")
                item["max_grasp_force"] = max(item["max_grasp_force"], float(grasp_forces[eid]))
                item["max_error"] = max(item["max_error"], float(np.max(np.abs(actual-plan.q[index]))))
                item["max_gripper_error"] = max(item["max_gripper_error"], float(np.max(np.abs(fingers-gripper))))
                transform = UsdGeom.Xformable(world.stage.GetPrimAtPath(env.link6_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                tip = np.array(transform.Transform(Gf.Vec3d(0, 0, cfg["tip_offset"])))
                if not np.isfinite(tip).all():
                    raise RuntimeError(f"Nonfinite world tool transform in episode {item['id']} step {step}")
                fk_error = np.linalg.norm(tip - (kin.fk(actual)[:3, 3] + base + env.offset))
                item["max_fk_error"] = max(item["max_fk_error"], float(fk_error))
                button_positions = np.array([env.buttons[f].get_world_pose()[0] for f in range(24,36)])
                rest_positions = np.array([env.button_info[f]["body_center"] for f in range(24,36)])
                lateral_error = float(np.max(np.abs(button_positions[:, 1:] - rest_positions[:, 1:])))
                item["max_button_lateral_error"] = max(item["max_button_lateral_error"], lateral_error)
                if not np.isfinite(lateral_error) or lateral_error > .0002:
                    raise RuntimeError(f"Panel button escaped its shifted spring anchor: {lateral_error}")
                travel = button_positions[:, 0] - rest_positions[:, 0]
                if not np.isfinite(travel).all():
                    raise RuntimeError(f"Nonfinite button travel in episode {item['id']} step {step}")
                if step == 0:
                    velocity = np.asarray(env.robot.get_joint_velocities())[arm_i]
                    if not np.isfinite(velocity).all():
                        raise RuntimeError(f"Nonfinite initial velocity in episode {item['id']}")
                    item["initial_home_error"] = float(np.max(np.abs(actual - home)))
                    item["initial_joint_velocity"] = velocity.tolist()
                    item["initial_max_speed"] = float(np.max(np.abs(velocity)))
                    item["initial_button_travel"] = travel.tolist()
                    item["initial_max_button_travel"] = float(np.max(np.abs(travel)))
                for f in range(24,36):
                    j = f - 24
                    on = f not in item["lit"] and travel[j] >= cfg["press_threshold"] and forces[eid,j] > .02
                    off = f in item["lit"] and travel[j] <= cfg["release_threshold"]
                    if on or off:
                        light_changed_since_capture = True
                        item["lit"].add(f) if on else item["lit"].remove(f)
                        set_button_light(world.stage, env.button_info[f], on)
                        item["events"].append(dict(type="pressed" if on else "released", floor=f,
                            physics_index=step, time=step*dt, travel_m=float(travel[j]), force_n=float(forces[eid,j])))
                lights = np.array([int(f in item["lit"]) for f in range(24,36)], dtype=np.uint8)
                item["physics"].append((actual.copy(), plan.q[index].copy(), fingers.copy(), travel, forces[eid].copy(), lights))
                if step % stride == 0:
                    state, action = sample_state_action(kin, plan, actual, fingers, index, stride)
                    future = min(index+stride, len(plan.q)-1)
                    item["frames"].append((state, action, step*dt, actual.copy(), plan.q[future].copy(),
                                           step, str(plan.phase[index]), lights.copy()))
                    captures.append(eid)
            if step % stride == 0:
                # Request a synchronous annotated frame at the already measured
                # physics state. Ordinary app updates can return the previous
                # tiled RGB capture even when repeated without a physics step.
                time_before_capture = world.current_time
                subframes = LIGHT_SETTLE_SUBFRAMES if light_changed_since_capture else 2
                if light_changed_since_capture:
                    import omni.usd
                    omni.usd.get_context().reset_renderer_accumulation()
                    # Some RTX histories advance per capture rather than per
                    # subframe. Settle material changes at the same physics time.
                    for _ in range(LIGHT_SETTLE_CAPTURES):
                        rep.orchestrator.step(delta_time=0., pause_timeline=False, rt_subframes=2)
                rep.orchestrator.step(delta_time=0., pause_timeline=False, rt_subframes=subframes)
                light_changed_since_capture = False
                if abs(world.current_time - time_before_capture) > 1e-10:
                    raise RuntimeError("RGB capture unexpectedly advanced physics time")
                for eid in captures:
                    measured_after = envs[eid].robot.get_joint_positions()[controls[eid][0]]
                    if not np.array_equal(measured_after, batch[eid]["frames"][-1][3]):
                        raise RuntimeError("RGB capture unexpectedly advanced the robot state")
                tiles = split_rgb(annotator.get_data())
                for eid in captures:
                    item = batch[eid]
                    for view in range(2):
                        rgb = tiles[2*eid+view]
                        if rgb.shape != (480,640,3) or rgb.std() < 2:
                            raise RuntimeError(f"Empty RGB episode {item['id']} camera {view}")
                        item["videos"][view].add(rgb)
                        if step == 0 or (item["lit"] and not (item["directory"] / f"press_{view}.jpg").exists()):
                            Image.fromarray(rgb).save(item["directory"] / f"{'initial' if step == 0 else 'press'}_{view}.jpg")
            if step % 600 == 0:
                print(f"SIM batch={batch_start} step={step}/{nsteps}", flush=True)
        for eid, item in enumerate(batch):
            for video in item["videos"]:
                video.close()
            frames, physics = item["frames"], item["physics"]
            np.savez_compressed(item["directory"] / "frames.npz",
                **{key: np.asarray([r[j] for r in frames]) for j,key in enumerate(
                    ("state", "action", "sim_time", "q_actual", "q_target", "physics_index", "phase", "lights"))})
            np.savez_compressed(item["directory"] / "physics.npz",
                **{key: np.asarray([r[j] for r in physics]) for j,key in enumerate(
                    ("q_actual", "q_command", "gripper_actual", "button_travel", "contact_force", "lights"))})
            home_error = float(np.max(np.abs(physics[-1][0]-home)))
            presses = [e["floor"] for e in item["events"] if e["type"] == "pressed"]
            releases = [e["floor"] for e in item["events"] if e["type"] == "released"]
            success = (presses == releases == [item["floor"]] and not item["lit"] and not unexpected[eid]
                       and home_error < cfg["home_tolerance_rad"] and item["max_error"] < .15
                       and item["max_fk_error"] < .005 and item["max_gripper_error"] < .00025
                       and item["initial_home_error"] < cfg["home_tolerance_rad"]
                       and item["initial_max_speed"] < START_SPEED_TOLERANCE_RAD_S
                       and item["initial_max_button_travel"] <= cfg["release_threshold"]
                       and all(v.frames == len(frames) for v in item["videos"]))
            metadata = dict(episode_id=item["id"], floor=item["floor"], task=f"Press {item['floor']} floor.",
                seed=item["seed"], success=success, fps=fps, num_frames=len(frames), physics_steps=len(physics),
                physics_hz=120, capture_stride=stride, action_horizon_s=1 / fps,
                raw_schema_version=manifest["raw_schema_version"], collection_fingerprint=manifest["collection_fingerprint"],
                plan_physics_steps=len(item["plan"].q),
                pose_frame="base_link", pose_link="gripper_tcp", tcp_offset_link6_m=[0,0,.1358],
                press_tip_offset_link6_m=[0,0,.24], gripper_width_m=.008, env_offset_m=envs[eid].offset.tolist(),
                robot_base_world_m=(base+envs[eid].offset).tolist(), events=item["events"],
                sample_time_origin="first captured post-physics state", first_sample_physics_time_s=dt,
                max_intended_grasp_force_n=item["max_grasp_force"],
                max_joint_error_rad=item["max_error"], max_fk_error_m=item["max_fk_error"],
                max_gripper_error_m=item["max_gripper_error"], final_home_error_rad=home_error,
                initial_home_error_rad=item["initial_home_error"],
                initial_joint_velocity_rad_s=item["initial_joint_velocity"],
                initial_max_joint_speed_rad_s=item["initial_max_speed"],
                initial_joint_speed_tolerance_rad_s=START_SPEED_TOLERANCE_RAD_S,
                initial_button_travel_m=item["initial_button_travel"],
                initial_max_abs_button_travel_m=item["initial_max_button_travel"],
                panel_offset_y_m=item["plan"].metadata["panel_offset_y_m"],
                panel_offset_x_m=item["plan"].metadata["panel_offset_x_m"],
                panel_layout=item["panel_layout"],
                max_button_lateral_error_m=item["max_button_lateral_error"],
                unexpected_collisions=unexpected[eid], variation=item["plan"].metadata)
            from pressb.panel_metadata import validate_episode_panel_metadata
            validate_episode_panel_metadata(manifest, metadata)
            validate_source_arrays(item["directory"], metadata)
            for view in ("wrist", "global"):
                validate_encoded_video(item["directory"] / f"{view}.mp4", len(frames), fps)
            metadata["source_files"] = {name: file_identity(item["directory"] / name) for name in SOURCE_NAMES}
            write_json(item["directory"] / "metadata.json", metadata)
            if not success:
                failed = args.output / f".episode_{item['id']:06d}.failed_{time.time_ns()}"
                item["directory"].rename(failed)
                raise RuntimeError(f"Episode {item['id']} failed physical validation: {failed}")
            item["directory"].rename(args.output / f"episode_{item['id']:06d}")
            completed += 1
            print(f"COMMIT episode={item['id']} floor={item['floor']} frames={len(frames)} home={home_error:.6g}", flush=True)
        elapsed = time.monotonic()-start_wall
        status.update(completed_this_run=completed, elapsed_collection_seconds=elapsed,
                      episodes_per_hour=completed/max(elapsed,1)*3600,
                      estimated_remaining_seconds=(len(todo)-completed)*elapsed/max(completed,1))
        write_json(status_path, status)


if __name__ == "__main__":
    main()
