#!/usr/bin/env python3
"""Execute recorded base-frame TCP actions in a fresh, physical Isaac scene.

Prepare inputs with prepare_replay.py in the LeRobot environment first. Only
action and a single initial joint state enter the controller. Recorded joint
trajectories and recorded light events are never used to drive the simulator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from collect_dataset import (
    DATASET_ANTI_ALIASING, LIGHT_SETTLE_CAPTURES, LIGHT_SETTLE_SUBFRAMES,
    RENDER_EXPOSURE_CONTROLS, RENDER_HISTORY_CONTROLS, VideoPipe,
    capture_stride_for, file_identity, validate_encoded_video, write_json,
)
from prepare_replay import FULL_CYCLE_END, PRESS_ONLY_END, episode_end_policy


def terminal_condition_met(episode_end, floor, presses, releases, lit, final_home_error,
                          terminal_travel, cfg):
    """Use the source dataset's endpoint; preserve legacy full-cycle semantics."""
    if episode_end == PRESS_ONLY_END:
        return presses == [floor] and not releases and lit == {floor}
    if episode_end == FULL_CYCLE_END:
        return (presses == releases == [floor] and not lit
                and final_home_error < cfg["home_tolerance_rad"]
                and max(abs(float(value)) for value in terminal_travel) <= cfg["release_threshold"])
    raise ValueError(f"Unsupported recorded episode_end: {episode_end}")


def prepare_controls(args, cfg):
    import numpy as np
    from pressb.replay_control import solve_action_trajectory

    manifest = json.loads((args.input / "manifest.json").read_text())
    collection_path = args.dataset / "meta/collection_metadata.json"
    collection = json.loads(collection_path.read_text())
    export_path = args.dataset / "meta/export_manifest.json"
    exported = json.loads(export_path.read_text())
    export_hash = file_identity(export_path)["sha256"]
    episode_end = episode_end_policy(exported["semantics"])
    if manifest["semantics"] != exported["semantics"]:
        raise ValueError("Prepared action semantics differ from the recorded export")
    if manifest.get("episode_end", FULL_CYCLE_END) != episode_end:
        raise ValueError("Prepared episode termination differs from the recorded export")
    if cfg != collection["config"]:
        raise ValueError("Replay config must equal the recorded collection config")
    if file_identity(args.snapshot)["sha256"] != collection["scene_sha256"]:
        raise ValueError("Replay snapshot differs from the recorded scene")
    if (manifest["dataset"]["export_manifest_sha256"] != file_identity(args.dataset / "meta/export_manifest.json")["sha256"]
            or manifest["collection_metadata"]["sha256"] != file_identity(collection_path)["sha256"]):
        raise ValueError("Prepared actions belong to a different dataset or collection")
    fps = collection["fps"]
    stride = capture_stride_for(fps, cfg["physics_dt"])
    if (collection["pose_frame"] != "base_link" or collection["pose_link"] != "gripper_tcp"
            or collection["tcp_offset_link6_m"] != [0, 0, .1358]
            or collection["action_horizon_s"] != 1 / fps):
        raise ValueError("Unsupported recorded action semantics")
    trajectories = []
    paths = sorted(args.input.glob("episode_*.npz"))
    if not paths:
        raise ValueError("No prepared replay episodes")
    entries = {entry["source_episode_id"]: entry for entry in manifest["episodes"]}
    if len(entries) != len(paths) or {int(path.stem.split("_")[-1]) for path in paths} != set(entries):
        raise ValueError("Prepared input files differ from the selection manifest")
    (args.output / "controls").mkdir()
    for path in paths:
        meta = json.loads(path.with_suffix(".json").read_text())
        source_id = int(path.stem.split("_")[-1])
        entry = entries[source_id]
        if (file_identity(path)["sha256"] != entry["npz_sha256"]
                or file_identity(path.with_suffix(".json"))["sha256"] != entry["manifest_sha256"]
                or meta["source_episode_id"] != source_id):
            raise ValueError(f"Prepared input provenance mismatch: {path}")
        if meta.get("episode_end", FULL_CYCLE_END) != episode_end:
            raise ValueError("Prepared episode termination differs from the recorded export")
        policy_source = meta.get("episode_end_source_export_manifest_sha256")
        if (policy_source is not None or episode_end == PRESS_ONLY_END) and policy_source != export_hash:
            raise ValueError("Episode termination is not bound to the recorded export manifest")
        with np.load(path, allow_pickle=False) as data:
            action = data["action"].copy()
            state = data["state"].copy()
            seed = data["initial_joint_position"].copy()
            sample_times = data["sim_time"].copy()
        if (action.ndim != 2 or action.shape[1] != 8 or state.shape != action.shape
                or len(action) != entry["frames"]):
            raise ValueError(f"Invalid action/state arrays: {path}")
        if not np.allclose(sample_times, np.arange(len(action)) / fps, atol=1e-6, rtol=0):
            raise ValueError(f"Invalid prepared sample times: {path}")
        print(f"IK source={source_id} targets={len(action)}", flush=True)
        plan = solve_action_trajectory(ROOT / cfg["robot_urdf"], action, seed,
            fps=fps, physics_hz=120, initial_gripper_width=float(state[0, 7]))
        control_path = args.output / "controls" / path.name
        np.savez_compressed(control_path, joint_positions=plan.joint_positions,
            gripper_widths=plan.gripper_widths, times_s=plan.times_s,
            ik_solutions=plan.ik_solutions, position_errors_m=plan.position_errors_m,
            rotation_errors_rad=plan.rotation_errors_rad)
        floor = int(meta["floor"])
        if meta["task"] != f"Press {floor} floor." or not 24 <= floor <= 35:
            raise ValueError(f"Unexpected task: {meta}")
        if collection["raw_schema_version"] >= 11 and any(key not in meta for key in ("panel_offset_x_m", "panel_offset_y_m")):
            raise ValueError("Randomized replay input is missing its recorded panel offset")
        panel_offset = float(meta.get("panel_offset_y_m", cfg.get("panel_offset_y_m", 0.)))
        panel_offset_x = float(meta.get("panel_offset_x_m", cfg.get("panel_offset_x_m", 0.)))
        if not np.isfinite([panel_offset, panel_offset_x]).all():
            raise ValueError("Recorded panel offset must be finite")
        trajectories.append({"id": source_id, "floor": floor, "task": meta["task"],
            "episode_end": episode_end,
            "episode_end_source_export_manifest_sha256": export_hash,
            "panel_offset_y_m": panel_offset,
            "panel_offset_x_m": panel_offset_x,
            "input_metadata": meta, "input_file": str(path), "input_sha256": file_identity(path)["sha256"],
            "control_sha256": file_identity(control_path)["sha256"], "plan": plan,
            "recorded_state": state, "recorded_action": action})
    write_json(args.output / "replay_manifest.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "dataset": str(args.dataset),
        "input": str(args.input), "input_manifest_sha256": file_identity(args.input / "manifest.json")["sha256"],
        "export_manifest_sha256": file_identity(args.dataset / "meta/export_manifest.json")["sha256"],
        "collection_fingerprint": collection["collection_fingerprint"], "config": cfg,
        "episode_end": episode_end, "episode_end_source_export_manifest_sha256": export_hash,
        "scene": str(args.snapshot), "scene_sha256": collection["scene_sha256"],
        "fps": fps, "physics_hz": 120, "capture_stride": stride, "action_horizon_s": 1 / fps,
        "control": "recorded 8D TCP action -> continuous bounded full-pose IK -> 120 Hz joint interpolation",
        "initialization": "recorded first joint state once per episode, followed by physical settling",
        "recorded_joint_targets_used_for_control": False, "recorded_light_events_used": False,
        "source_episode_ids": [item["id"] for item in trajectories],
        "source_code_sha256": {name: file_identity(ROOT / name)["sha256"] for name in (
            "scripts/replay_dataset.py", "scripts/prepare_replay.py", "src/pressb/replay_control.py",
            "src/pressb/dataset_scene.py", "src/pressb/kinematics.py", "src/pressb/scene.py")},
        "prepared_manifest": manifest,
    })
    return trajectories, fps, stride


def simulate(app, args, cfg, items, fps, stride, status):
    import numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from pxr import Gf, Usd, UsdGeom, UsdLux, PhysicsSchemaTools
    import carb
    import omni.physx
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.utils.types import ArticulationAction
    from pressb.dataset_scene import create_envs, build_tiled_rgb, set_light, set_panel_offset, validate_panel_layout
    from pressb.kinematics import PiperKinematics

    dt = float(cfg["physics_dt"])
    home = np.asarray(cfg["home_q"])
    tcp_kin = PiperKinematics(ROOT / cfg["robot_urdf"], tip_offset=.1358)
    tip_kin = PiperKinematics(ROOT / cfg["robot_urdf"], tip_offset=.24)
    world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=1 / fps,
                  backend="numpy", device="cpu")
    world.get_physics_context().set_physx_update_transformations_settings(
        update_to_usd=True, update_velocities_to_usd=True)
    envs = create_envs(world, args.snapshot, min(args.num_envs, len(items)), cfg=cfg)
    camera_paths = [path for env in envs for path in (env.wrist_camera_path, env.global_camera_path)]
    product, annotator, split_rgb = build_tiled_rgb(camera_paths)
    rep.orchestrator.set_capture_on_play(False)
    world.reset()
    controls = []
    for env in envs:
        robot = env.robot
        arm = np.array([robot.get_dof_index(name) for name in tcp_kin.joint_names])
        fingers = np.array([robot.get_dof_index(name) for name in ("joint7", "joint8")])
        command = robot.get_joint_positions().copy()
        command[arm], command[fingers] = home, cfg["gripper_joint_positions_m"]
        robot.set_joint_positions(command)
        robot.set_joint_velocities(np.zeros_like(command))
        controller = robot.get_articulation_controller()
        controller.set_gains(kps=np.full(len(command), 12000.), kds=np.full(len(command), 600.))
        controller.apply_action(ArticulationAction(joint_positions=command))
        controls.append((arm, fingers, command, controller))
    lighting = {"domes": [{"path": str(prim.GetPath()), "active": prim.IsActive()}
        for prim in world.stage.TraverseAll() if prim.IsA(UsdLux.DomeLight)]}
    if sum(row["active"] for row in lighting["domes"]) != 1:
        raise RuntimeError("Replay requires exactly one active Dome")
    write_json(args.output / "lighting.json", lighting)
    settings = carb.settings.get_settings()
    force = np.zeros((len(envs), 12))
    unintended = [[] for _ in envs]
    step_now = [-1]
    body_map = {info["body_path"]: (env.env_id, floor)
                for env in envs for floor, info in env.button_info.items()}
    tool_paths = {path for env in envs for path in env.tool_colliders}

    def on_contact(headers, data):
        for header in headers:
            if not header.num_contact_data:
                continue
            actors = [str(PhysicsSchemaTools.intToSdfPath(x)) for x in (header.actor0, header.actor1)]
            colliders = [str(PhysicsSchemaTools.intToSdfPath(x)) for x in (header.collider0, header.collider1)]
            pair = next((body_map[actor] for actor in actors if actor in body_map), None)
            magnitude = sum(float(np.linalg.norm(data[k].impulse)) / dt
                for k in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data))
            if pair is not None and any(collider in tool_paths for collider in colliders):
                force[pair[0], pair[1] - 24] += magnitude
                continue
            for env in envs:
                prefix = env.robot_path + "/"
                robot_actors = [actor for actor in actors if actor.startswith(prefix)]
                if not robot_actors or magnitude <= .1:
                    continue
                shaft = prefix + "link6/PressStylus"
                pads = {prefix + "link7/collisions", prefix + "link8/collisions"}
                if shaft in colliders and any(collider in pads for collider in colliders):
                    continue
                reason = None
                if pair is not None:
                    reason = "non_stylus_button_contact"
                elif len(robot_actors) == 2 and actors[0] != actors[1]:
                    reason = "robot_self_contact"
                elif len(robot_actors) == 1 and robot_actors[0].rsplit("/", 1)[-1] not in ("dummy_link", "base_link", "link1"):
                    reason = "robot_environment_contact"
                if reason and len(unintended[env.env_id]) < 100:
                    unintended[env.env_id].append({"step": step_now[0], "actors": actors,
                        "colliders": colliders, "force_n": magnitude, "reason": reason})

    subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    completed = []
    for offset in range(0, len(items), len(envs)):
        batch = items[offset:offset + len(envs)]
        unintended = [[] for _ in envs]
        step_now[0] = -1
        for eid, item in enumerate(batch):
            set_panel_offset(world, envs[eid], item["panel_offset_y_m"], offset_x_m=item["panel_offset_x_m"])
            arm, fingers, command, controller = controls[eid]
            plan = item["plan"]
            command[arm] = plan.joint_positions[0]
            command[fingers] = [plan.gripper_widths[0] / 2, -plan.gripper_widths[0] / 2]
            # Reset only the episode's initial condition, never the moving arm.
            envs[eid].robot.set_joint_positions(command)
            envs[eid].robot.set_joint_velocities(np.zeros_like(command))
            controller.apply_action(ArticulationAction(joint_positions=command))
            for floor in range(24, 36):
                set_light(envs[eid], floor, False)
        for step in range(90):
            force.fill(0.)
            world.step(render=False)
            if step % 4 == 0:
                world.render()
        for _ in range(8):
            world.render()
        for key, value in {**RENDER_HISTORY_CONTROLS, **RENDER_EXPOSURE_CONTROLS}.items():
            settings.set(key, value)
        initial_time = world.current_time
        for attempt in range(12):
            rep.orchestrator.step(delta_time=0., pause_timeline=False, rt_subframes=2)
            if abs(world.current_time - initial_time) > 1e-10:
                raise RuntimeError("Rendering advanced physics during warmup")
            try:
                tiles = split_rgb(annotator.get_data())
                break
            except ValueError:
                if attempt == 11:
                    raise
        for eid, item in enumerate(batch):
            item["panel_layout"] = validate_panel_layout(world, envs[eid])
            directory = args.output / f"episode_{item['id']:06d}"
            directory.mkdir()
            item.update(directory=directory, physics=[], frames=[], events=[], lit=set(),
                captures=[], initial_velocity=envs[eid].robot.get_joint_velocities()[controls[eid][0]].tolist(),
                videos=[VideoPipe(directory / f"{view}.mp4", fps) for view in ("wrist", "global")])
        count = max(len(item["plan"].joint_positions) for item in batch)
        print(f"REPLAY batch={offset} source_ids={[item['id'] for item in batch]} steps={count}", flush=True)
        changed = False
        try:
            for step in range(count):
                if not app.is_running():
                    raise RuntimeError("Simulator closed before replay completed")
                for eid, item in enumerate(batch):
                    plan = item["plan"]
                    index = min(step, len(plan.joint_positions) - 1)
                    arm, fingers, command, controller = controls[eid]
                    command[arm] = plan.joint_positions[index]
                    command[fingers] = [plan.gripper_widths[index] / 2, -plan.gripper_widths[index] / 2]
                    controller.apply_action(ArticulationAction(joint_positions=command))
                force.fill(0.)
                step_now[0] = step
                world.step(render=False)
                captures = []
                for eid, item in enumerate(batch):
                    plan = item["plan"]
                    if step >= len(plan.joint_positions):
                        continue
                    env = envs[eid]
                    arm, fingers, command, _ = controls[eid]
                    actual = env.robot.get_joint_positions()[arm].copy()
                    actual_fingers = env.robot.get_joint_positions()[fingers].copy()
                    travel = np.array([env.buttons[floor].get_world_pose()[0][0] - env.button_info[floor]["rest_x"]
                                       for floor in range(24, 36)])
                    for floor in range(24, 36):
                        on = floor not in item["lit"] and travel[floor - 24] >= cfg["press_threshold"] and force[eid, floor - 24] > .02
                        off = floor in item["lit"] and travel[floor - 24] <= cfg["release_threshold"]
                        if on or off:
                            changed = True
                            item["lit"].add(floor) if on else item["lit"].remove(floor)
                            set_light(env, floor, on)
                            item["events"].append({"type": "pressed" if on else "released", "floor": floor,
                                "physics_index": step, "time": step * dt, "travel_m": float(travel[floor - 24]),
                                "force_n": float(force[eid, floor - 24])})
                    lights = np.array([floor in item["lit"] for floor in range(24, 36)], dtype=np.uint8)
                    tcp = tcp_kin.fk(actual)
                    quat = Rotation.from_matrix(tcp[:3, :3]).as_quat()[[3, 0, 1, 2]]
                    measured_state = np.r_[tcp[:3, 3], quat, actual_fingers[0] - actual_fingers[1]]
                    transform = UsdGeom.Xformable(world.stage.GetPrimAtPath(env.link6_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    tip_world = np.asarray(transform.Transform(Gf.Vec3d(0, 0, .24)))
                    record = (actual, plan.joint_positions[step].copy(), actual_fingers,
                        command[fingers].copy(), travel, force[eid].copy(), lights, measured_state,
                        tip_world.copy(), step * dt)
                    if not all(np.isfinite(array).all() for array in record):
                        raise RuntimeError(f"Nonfinite physics in replay {item['id']} at {step}")
                    item["physics"].append(record)
                    if step % stride == 0:
                        frame_index = step // stride
                        item["frames"].append((measured_state.copy(), item["recorded_action"][frame_index].copy(),
                                               step * dt, step, actual.copy(), lights.copy()))
                        item["captures"].append(step)
                        captures.append(eid)
                if captures:
                    before = world.current_time
                    if changed:
                        omni.usd.get_context().reset_renderer_accumulation()
                        for _ in range(LIGHT_SETTLE_CAPTURES):
                            rep.orchestrator.step(delta_time=0., pause_timeline=False, rt_subframes=2)
                    rep.orchestrator.step(delta_time=0., pause_timeline=False,
                        rt_subframes=LIGHT_SETTLE_SUBFRAMES if changed else 2)
                    changed = False
                    if abs(world.current_time - before) > 1e-10:
                        raise RuntimeError("Camera capture advanced physics")
                    tiles = split_rgb(annotator.get_data())
                    for eid in captures:
                        item = batch[eid]
                        if not np.array_equal(envs[eid].robot.get_joint_positions()[controls[eid][0]], item["frames"][-1][4]):
                            raise RuntimeError("Camera capture changed robot state")
                        for view, name in enumerate(("wrist", "global")):
                            rgb = tiles[2 * eid + view]
                            if rgb.shape != (480, 640, 3) or rgb.std() < 2:
                                raise RuntimeError(f"Invalid replay RGB source={item['id']} camera={name}")
                            item["videos"][view].add(rgb)
                            suffix = "initial" if step == 0 else "pressed" if item["lit"] else "released"
                            if suffix != "released" or any(event["type"] == "released" for event in item["events"]):
                                image_path = item["directory"] / f"{name}_{suffix}.jpg"
                                if not image_path.exists():
                                    Image.fromarray(rgb).save(image_path)
                            if step == len(item["plan"].joint_positions) - 1:
                                Image.fromarray(rgb).save(item["directory"] / f"{name}_final.jpg")
                if step % 600 == 0:
                    print(f"REPLAY_PROGRESS batch={offset} step={step}/{count}", flush=True)
        finally:
            for item in batch:
                for video in item["videos"]:
                    video.close()
        for eid, item in enumerate(batch):
            physics = {key: np.asarray([row[index] for row in item["physics"]]) for index, key in enumerate(
                ("q_actual", "q_command", "gripper_actual", "gripper_command", "button_travel",
                 "contact_force", "lights", "tcp_state", "tip_position_world", "time"))}
            frames = {key: np.asarray([row[index] for row in item["frames"]]) for index, key in enumerate(
                ("state", "action", "sim_time", "physics_index", "q_actual", "lights"))}
            np.savez_compressed(item["directory"] / "physics.npz", **physics)
            np.savez_compressed(item["directory"] / "frames.npz", **frames)
            presses = [event["floor"] for event in item["events"] if event["type"] == "pressed"]
            releases = [event["floor"] for event in item["events"] if event["type"] == "released"]
            max_joint_error = float(np.max(np.abs(physics["q_actual"] - physics["q_command"])))
            max_gripper_error = float(np.max(np.abs(physics["gripper_actual"] - physics["gripper_command"])))
            final_home_error = float(np.max(np.abs(physics["q_actual"][-1] - home)))
            max_fk_error = max(float(np.linalg.norm(tip_kin.fk(q)[:3, 3] + envs[eid].base_position - tip))
                for q, tip in zip(physics["q_actual"], physics["tip_position_world"]))
            terminal_ok = terminal_condition_met(item["episode_end"], item["floor"], presses, releases,
                item["lit"], final_home_error, physics["button_travel"][-1], cfg)
            success = (terminal_ok and not unintended[eid]
                and max_joint_error < .15 and max_gripper_error < .00025 and max_fk_error < .005)
            for name in ("wrist", "global"):
                validate_encoded_video(item["directory"] / f"{name}.mp4", len(item["frames"]), fps)
            metadata = {"source_episode_id": item["id"], "floor": item["floor"], "task": item["task"],
                "episode_end": item["episode_end"],
                "episode_end_source_export_manifest_sha256": item["episode_end_source_export_manifest_sha256"],
                "panel_offset_x_m": item["panel_offset_x_m"],
                "panel_offset_y_m": item["panel_offset_y_m"], "panel_layout": item["panel_layout"],
                "success": success, "config": cfg, "fps": fps, "physics_hz": 120,
                "capture_stride": stride, "action_horizon_s": 1 / fps,
                "physics_steps": len(item["physics"]), "num_frames": len(item["frames"]),
                "capture_physics_indices": item["captures"], "events": item["events"],
                "unexpected_collisions": unintended[eid], "max_joint_error_rad": max_joint_error,
                "max_gripper_error_m": max_gripper_error, "max_fk_error_m": max_fk_error,
                "final_home_error_rad": final_home_error,
                "initial_joint_velocity_rad_s": item["initial_velocity"],
                "initial_home_error_rad": float(np.max(np.abs(physics["q_actual"][0] - home))),
                "initial_max_abs_button_travel_m": float(np.max(np.abs(physics["button_travel"][0]))),
                "pose_frame": "base_link", "pose_link": "gripper_tcp", "tcp_offset_link6_m": [0, 0, .1358],
                "robot_base_world_m": envs[eid].base_position.tolist(), "env_offset_m": envs[eid].offset.tolist(),
                "input_file": item["input_file"], "input_sha256": item["input_sha256"],
                "control_sha256": item["control_sha256"],
                "max_ik_position_error_m": float(np.max(item["plan"].position_errors_m)),
                "max_ik_rotation_error_rad": float(np.max(item["plan"].rotation_errors_rad)),
                "initial_ik_seed_clipped_to_joint_limits": item["plan"].initial_seed_clipped,
                "recorded_state_position_difference_max_m": float(np.max(np.linalg.norm(
                    frames["state"][:, :3] - item["recorded_state"][:, :3], axis=1))),
                "recorded_joint_targets_used_for_control": False,
                "source_files": {name: file_identity(item["directory"] / name) for name in
                                 ("physics.npz", "frames.npz", "wrist.mp4", "global.mp4")}}
            write_json(item["directory"] / "metadata.json", metadata)
            completed.append({key: metadata[key] for key in ("source_episode_id", "floor", "success", "num_frames",
                "max_joint_error_rad", "final_home_error_rad", "max_ik_position_error_m", "max_ik_rotation_error_rad")})
            print(f"REPLAY_RESULT {json.dumps(completed[-1])}", flush=True)
            del item["physics"], item["frames"]
        status.update(status="replaying", completed_episodes=len(completed),
                      passed_episodes=sum(row["success"] for row in completed))
        write_json(args.output / "status.json", status)
    report = {"success": all(row["success"] for row in completed), "total_episodes": len(completed),
              "passed_episodes": sum(row["success"] for row in completed), "fps": fps,
              "episodes": completed, "finished_at": datetime.now(timezone.utc).isoformat()}
    write_json(args.output / "report.json", report)
    return report["success"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/replay_edge_30hz/input")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/piper_elevator_lerobot_edge_30hz")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "outputs/edge30_source/scene.usda")
    parser.add_argument("--config", type=Path, default=ROOT / "outputs/edge30_source/config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/replay_edge_30hz/run_v1")
    parser.add_argument("--num-envs", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=4)
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("num-envs must be positive")
    for name in ("input", "dataset", "snapshot", "config", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text())
    status = {"status": "preparing", "started_at": datetime.now(timezone.utc).isoformat(),
              "gpu": args.gpu, "num_envs": args.num_envs, "completed_episodes": 0}
    app = None
    exit_code = 1
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    try:
        write_json(args.output / "status.json", status)
        items, fps, stride = prepare_controls(args, cfg)
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "create_new_stage": False, "width": 640, "height": 480,
            "active_gpu": args.gpu, "physics_gpu": args.gpu, "multi_gpu": False,
            "renderer": "RayTracedLighting", "anti_aliasing": DATASET_ANTI_ALIASING, "fast_shutdown": True,
            "extra_args": ["--/app/asyncRendering=false", "--/rtx/ecoMode/enabled=false",
                "--/plugins/carb.tasking.plugin/threadCount=8", "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8",
                "--/persistent/physics/numThreads=4", "--/validate/p2p/enabled=false", "--/validate/iommu/enabled=false"]})
        success = simulate(app, args, cfg, items, fps, stride, status)
        status.update(status="complete" if success else "failed_physics", finished_at=datetime.now(timezone.utc).isoformat())
        exit_code = 0 if success else 1
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      finished_at=datetime.now(timezone.utc).isoformat())
        traceback.print_exc()
    finally:
        write_json(args.output / "status.json", status)
        if app is not None:
            app.app.post_quit(exit_code)
            app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
