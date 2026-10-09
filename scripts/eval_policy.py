#!/usr/bin/env python3
"""Closed-loop VLA evaluation from live Isaac cameras and measured TCP state.

Only a task sentence, measured state, and the two current RGB images reach the
model. Dataset metadata identifies the frozen scene; no demonstration actions,
expert plans, or target button coordinates are supplied to model or controller.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from collect_dataset import (
    DATASET_ANTI_ALIASING, LIGHT_SETTLE_CAPTURES, LIGHT_SETTLE_SUBFRAMES,
    RENDER_EXPOSURE_CONTROLS, RENDER_HISTORY_CONTROLS, VideoPipe,
    capture_stride_for, file_identity, validate_encoded_video, write_json,
)
from pressb.motion_smoothing import DEFAULT_SMOOTHING_WINDOW, smoothing_settings
from pressb.policy_layouts import (LAYOUT_FIELDS, evaluation_panel_layouts, evaluation_schedule,
                                   reposition_evaluation_panels, evaluation_layout_evidence)


def run(app, args, cfg, status):
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
    from pressb.dataset_scene import create_envs, build_tiled_rgb, set_light
    from pressb.kinematics import PiperKinematics
    from pressb.policy_eval import PolicyClient, PolicyPoseController, InvalidPolicyAction, validate_policy_reply
    from pressb.motion_smoothing import JointCommandSmoother
    from pressb.replay_control import action_pose

    fps, dt = 30, float(cfg["physics_dt"])
    stride = capture_stride_for(fps, dt)
    motion_smoothing = smoothing_settings(args.smoothing_window, physics_hz=round(1 / dt))
    max_steps = int(round(args.max_seconds * 120))
    home = np.asarray(cfg["home_q"])
    tcp_kin = PiperKinematics(ROOT / cfg["robot_urdf"], tip_offset=.1358)
    tip_kin = PiperKinematics(ROOT / cfg["robot_urdf"], tip_offset=.24)
    client = PolicyClient(args.endpoint, args.timeout)
    world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=1 / fps,
                  backend="numpy", device="cpu")
    world.get_physics_context().set_physx_update_transformations_settings(
        update_to_usd=True, update_velocities_to_usd=True)
    layouts = evaluation_panel_layouts(cfg, args.panel_layouts)
    schedule = evaluation_schedule(args.floors, args.episodes_per_floor, layouts)
    snapshot = args.snapshot
    if args.asset_bundle is not None:
        from pressb.scene_portability import prepare_runtime_snapshot
        relocation = prepare_runtime_snapshot(args.snapshot, args.asset_bundle, args.output)
        manifest_path = args.output / "eval_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["scene_relocation"] = relocation
        write_json(manifest_path, manifest)
        snapshot = Path(relocation["runtime_snapshot"])
    envs = create_envs(world, snapshot, min(args.num_envs, len(schedule)), cfg=cfg)
    camera_paths = [path for env in envs for path in (env.wrist_camera_path, env.global_camera_path)]
    product, annotator, split_rgb = build_tiled_rgb(camera_paths)
    rep.orchestrator.set_capture_on_play(False)
    world.reset()
    controls = []
    linear_commands = []
    for env in envs:
        arm = np.array([env.robot.get_dof_index(name) for name in tcp_kin.joint_names])
        fingers = np.array([env.robot.get_dof_index(name) for name in ("joint7", "joint8")])
        command = env.robot.get_joint_positions().copy()
        controller = env.robot.get_articulation_controller()
        controller.set_gains(kps=np.full(len(command), 12000.), kds=np.full(len(command), 600.))
        controls.append((arm, fingers, command, controller))
        linear_commands.append(command[arm].astype(np.float64))
    domes = [{"path": str(prim.GetPath()), "active": prim.IsActive()}
             for prim in world.stage.TraverseAll() if prim.IsA(UsdLux.DomeLight)]
    if sum(row["active"] for row in domes) != 1:
        raise RuntimeError("Policy evaluation requires exactly one active Dome")
    write_json(args.output / "lighting.json", {"domes": domes})
    settings = carb.settings.get_settings()
    for key, value in {**RENDER_HISTORY_CONTROLS, **RENDER_EXPOSURE_CONTROLS}.items():
        settings.set(key, value)
    forces = np.zeros((len(envs), 12))
    unintended = [[] for _ in envs]
    recording_envs = set()
    physics_now = [-1]
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
            magnitude = sum(float(np.linalg.norm(data[k].impulse)) / dt for k in
                            range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data))
            if pair is not None and any(collider in tool_paths for collider in colliders):
                forces[pair[0], pair[1] - 24] += magnitude
                continue
            for env in envs:
                if env.env_id not in recording_envs:
                    continue
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
                    unintended[env.env_id].append(dict(physics_index=physics_now[0], actors=actors,
                        colliders=colliders, force_n=magnitude, reason=reason))

    subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contact)

    def measured(eid):
        arm, fingers, command, _ = controls[eid]
        joints = envs[eid].robot.get_joint_positions().astype(np.float64)
        q, finger_q = joints[arm], joints[fingers]
        tcp = tcp_kin.fk(q)
        quat = Rotation.from_matrix(tcp[:3, :3]).as_quat()[[3, 0, 1, 2]]
        return np.r_[tcp[:3, 3], quat, finger_q[0] - finger_q[1]], q.copy(), finger_q.copy()

    def capture(changed=False):
        before = world.current_time
        prior = [env.robot.get_joint_positions().copy() for env in envs]
        if changed:
            omni.usd.get_context().reset_renderer_accumulation()
            for _ in range(LIGHT_SETTLE_CAPTURES):
                rep.orchestrator.step(delta_time=0., pause_timeline=False, rt_subframes=2)
        rep.orchestrator.step(delta_time=0., pause_timeline=False,
                              rt_subframes=LIGHT_SETTLE_SUBFRAMES if changed else 2)
        if abs(world.current_time - before) > 1e-10:
            raise RuntimeError("Camera capture advanced simulation time")
        if any(not np.array_equal(env.robot.get_joint_positions(), q) for env, q in zip(envs, prior)):
            raise RuntimeError("Camera capture changed robot state")
        result = split_rgb(annotator.get_data())
        if any(rgb.shape != (480, 640, 3) or rgb.std() < 2 for rgb in result):
            raise RuntimeError("Invalid live camera RGB")
        return result

    def record_physics(item, index, chunk=-1, action=-1, substep=0):
        eid, env = item["eid"], envs[item["eid"]]
        state, q, fingers = measured(eid)
        command = controls[eid][2]
        travel = np.array([env.buttons[floor].get_world_pose()[0][0] - env.button_info[floor]["rest_x"]
                           for floor in range(24, 36)])
        lights = np.array([floor in item["lit"] for floor in range(24, 36)], dtype=np.uint8)
        transform = UsdGeom.Xformable(world.stage.GetPrimAtPath(env.link6_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        tip_world = np.asarray(transform.Transform(Gf.Vec3d(0, 0, .24)))
        fk_error = float(np.linalg.norm(tip_world - tip_kin.fk(q)[:3, 3] - env.base_position))
        if fk_error > .005:
            raise RuntimeError(f"Measured FK differs from USD tool pose by {fk_error} m")
        # Goal coordinates are solely an evaluation metric, never policy inputs.
        distance = float(np.linalg.norm(tip_world - np.asarray(env.button_info[item["floor"]]["center"])))
        row = dict(q_actual=q, q_command=command[controls[eid][0]].copy(),
                   q_command_unsmoothed=linear_commands[eid].copy(), gripper_actual=fingers,
                   button_travel=travel, contact_force=forces[eid].copy(), lights=lights, state=state,
                   tip_position_world=tip_world, sim_time=index * dt, physics_index=index,
                   chunk_index=chunk, chunk_action_index=action, substep=substep,
                   target_tip_distance_m=distance, fk_error_m=fk_error)
        if not all(np.isfinite(value).all() for value in row.values()):
            raise RuntimeError("Nonfinite simulated state")
        item["physics"].append(row)
        return row

    def record_frame(item, tiles, terminal=False):
        row = item["physics"][-1]
        duplicate = bool(item["frames"] and item["frames"][-1]["physics_index"] == row["physics_index"])
        if not duplicate:
            item["frames"].append({key: row[key] for key in ("state", "q_actual", "lights", "sim_time", "physics_index")})
        for view, name in enumerate(("wrist", "global")):
            rgb = tiles[2 * item["eid"] + view]
            if not duplicate:
                item["videos"][view].add(rgb)
            if row["physics_index"] == 0:
                Image.fromarray(rgb).save(item["directory"] / f"{name}_initial.jpg")
            if terminal:
                Image.fromarray(rgb).save(item["directory"] / f"{name}_final.jpg")

    completed = []
    for offset in range(0, len(schedule), len(envs)):
        batch = []
        batch_schedule = schedule[offset:offset + len(envs)]
        recording_envs.clear()
        physics_now[0] = -1
        for eid, env in enumerate(envs):
            arm, fingers, command, controller = controls[eid]
            command[arm], command[fingers] = home, cfg["gripper_joint_positions_m"]
            linear_commands[eid] = command[arm].astype(np.float64)
            env.robot.set_joint_positions(command)
            env.robot.set_joint_velocities(np.zeros_like(command))
            controller.apply_action(ArticulationAction(joint_positions=command))
            for floor in range(24, 36):
                set_light(env, floor, False)
        # All arms are now at home. Apply complete panel/back-wall offsets and
        # spring anchors before the existing, unrecorded shared settling ticks.
        # Fixed mode leaves the legacy physical reset sequence unchanged.
        if args.panel_layouts != "fixed":
            reposition_evaluation_panels(world, envs, batch_schedule)
        for step in range(90):
            forces.fill(0.)
            world.step(render=False)
            if step % 4 == 0:
                world.render()
        for _ in range(8):
            world.render()
        panel_evidence = [evaluation_layout_evidence(world, envs[eid], episode)
                          for eid, episode in enumerate(batch_schedule)]
        unintended = [[] for _ in envs]
        forces.fill(0.)
        for attempt in range(12):
            try:
                tiles = capture(changed=True)
                break
            except ValueError:
                if attempt == 11:
                    raise
        for eid, scheduled in enumerate(batch_schedule):
            floor, repeat, episode_id = scheduled["floor"], scheduled["repeat"], scheduled["episode_id"]
            directory = args.output / f"episode_{episode_id:06d}"
            directory.mkdir()
            (directory / "observations").mkdir()
            (directory / "actions.jsonl").touch()
            _, q, _ = measured(eid)
            if np.max(np.abs(q - home)) > cfg["home_tolerance_rad"]:
                raise RuntimeError("Evaluation did not start at the collection home pose")
            item = dict(id=episode_id, eid=eid, floor=floor, repeat=repeat, task=f"Press {floor} floor.",
                seed_episode_index=scheduled["seed_episode_index"],
                **{key: scheduled[key] for key in LAYOUT_FIELDS}, panel_layout=panel_evidence[eid],
                directory=directory, lit=set(), events=[], physics=[], frames=[], actions=[], requests=[],
                done=False, termination=None, started_wall=time.monotonic(),
                initial_joint_velocity_rad_s=envs[eid].robot.get_joint_velocities()[controls[eid][0]].tolist(),
                solver=PolicyPoseController(ROOT / cfg["robot_urdf"], controls[eid][2][controls[eid][0]], fps=fps),
                smoother=JointCommandSmoother(linear_commands[eid], window=args.smoothing_window),
                videos=[VideoPipe(directory / f"{name}.mp4", fps) for name in ("wrist", "global")])
            row = record_physics(item, 0)
            if np.max(np.abs(row["button_travel"])) > cfg["release_threshold"]:
                raise RuntimeError("Button springs did not return before evaluation")
            record_frame(item, tiles)
            batch.append(item)
            recording_envs.add(eid)
        current_step, chunk = 0, 0
        try:
            while any(not item["done"] for item in batch):
                if not app.is_running():
                    raise RuntimeError("Simulator closed before evaluation completed")
                for item in batch:
                    if item["done"]:
                        continue
                    eid = item["eid"]
                    state, q, _ = measured(eid)
                    images = {"wrist": tiles[2 * eid], "global": tiles[2 * eid + 1]}
                    image_records = {}
                    for name, rgb in images.items():
                        path = item["directory"] / "observations" / f"chunk_{chunk:04d}_{name}.png"
                        Image.fromarray(rgb).save(path)
                        image_records[name] = dict(path=str(path.relative_to(item["directory"])),
                            rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), **file_identity(path))
                    # Keep each floor's random stream when evaluating a subset;
                    # the full ordered 12-floor schedule retains its old seeds.
                    seed = args.seed + item["seed_episode_index"] * 10000 + chunk
                    before = world.current_time
                    reply, wall_seconds = client.predict(item["task"], state, images, seed)
                    if abs(world.current_time - before) > 1e-10 or not np.array_equal(measured(eid)[1], q):
                        raise RuntimeError("Inference advanced simulation or changed measured joints")
                    request = dict(chunk_index=chunk, observation_physics_index=current_step,
                        observation_sim_time=current_step * dt, state=state.tolist(), q_actual=q.tolist(),
                        task=item["task"], seed=seed, images=image_records, response=reply,
                        wall_seconds=wall_seconds, frozen_sim_time=True)
                    item["requests"].append(request)
                    with (item["directory"] / "requests.jsonl").open("a") as handle:
                        handle.write(json.dumps(request, allow_nan=False) + "\n")
                    try:
                        item["chunk_pose8"], item["chunk_pose9"] = validate_policy_reply(reply)
                    except InvalidPolicyAction as exc:
                        item.update(done=True, termination="invalid_policy_action", error=str(exc))
                        recording_envs.discard(eid)
                        record_frame(item, tiles, terminal=True)
                for action_index in range(7):
                    active = [item for item in batch if not item["done"]]
                    if not active:
                        break
                    for item in active:
                        action = item["chunk_pose8"][action_index]
                        previous = item["solver"].q
                        target, diagnostics = item["solver"].solve(action)
                        item["pending_action"] = dict(chunk_index=chunk, chunk_action_index=action_index,
                            physics_start_index=current_step, physics_end_index=None,
                            action_pose8=action.tolist(), action_pose9=item["chunk_pose9"][action_index].tolist(),
                            q_before=previous.tolist(), q_target=target.tolist(),
                            q_executed_before=controls[item["eid"]][2][controls[item["eid"]][0]].tolist(),
                            **diagnostics)
                    for substep in range(1, stride + 1):
                        active = [item for item in batch if not item["done"]]
                        if not active:
                            break
                        current_step += 1
                        physics_now[0] = current_step
                        for item in active:
                            action = item["pending_action"]
                            arm, fingers, command, controller = controls[item["eid"]]
                            before_q, target_q = np.asarray(action["q_before"]), np.asarray(action["q_target"])
                            linear_commands[item["eid"]] = before_q + (target_q - before_q) * substep / stride
                            command[arm] = item["smoother"].step(linear_commands[item["eid"]])
                            command[fingers] = cfg["gripper_joint_positions_m"]
                            controller.apply_action(ArticulationAction(joint_positions=command))
                        forces.fill(0.)
                        world.step(render=False)
                        changed, terminal = False, []
                        for item in active:
                            eid, env = item["eid"], envs[item["eid"]]
                            travel = np.array([env.buttons[floor].get_world_pose()[0][0] - env.button_info[floor]["rest_x"]
                                               for floor in range(24, 36)])
                            for floor in range(24, 36):
                                on = floor not in item["lit"] and travel[floor - 24] >= cfg["press_threshold"] and forces[eid, floor - 24] > .02
                                off = floor in item["lit"] and travel[floor - 24] <= cfg["release_threshold"]
                                if on or off:
                                    changed = True
                                    item["lit"].add(floor) if on else item["lit"].remove(floor)
                                    set_light(env, floor, on)
                                    item["events"].append(dict(type="pressed" if on else "released", floor=floor,
                                        physics_index=current_step, sim_time=current_step * dt,
                                        travel_m=float(travel[floor - 24]), force_n=float(forces[eid, floor - 24])))
                            record_physics(item, current_step, chunk, action_index, substep)
                            presses = [event["floor"] for event in item["events"] if event["type"] == "pressed"]
                            reason = ("unexpected_collision" if unintended[eid] else
                                      "target_pressed" if presses == [item["floor"]] else
                                      "wrong_button_pressed" if presses else
                                      "time_limit" if current_step >= max_steps else None)
                            if reason:
                                item.update(done=True, termination=reason)
                                recording_envs.discard(eid)
                                terminal.append(item)
                            if substep == stride or reason:
                                item["pending_action"]["physics_end_index"] = current_step
                                # Record the actually submitted endpoint even if
                                # a contact terminates a partially executed interval.
                                executed_q = controls[eid][2][controls[eid][0]].astype(np.float64)
                                executed_tcp = tcp_kin.fk(executed_q)
                                raw_position, raw_rotation, _ = action_pose(item["pending_action"]["action_pose8"])
                                item["pending_action"].update(
                                    q_executed_endpoint=executed_q.tolist(),
                                    executed_position_residual_m=float(np.linalg.norm(executed_tcp[:3, 3] - raw_position)),
                                    executed_rotation_residual_rad=float(Rotation.from_matrix(
                                        raw_rotation @ executed_tcp[:3, :3].T).magnitude()))
                                item["actions"].append(item["pending_action"].copy())
                                with (item["directory"] / "actions.jsonl").open("a") as handle:
                                    handle.write(json.dumps(item["actions"][-1], allow_nan=False) + "\n")
                        if substep == stride or terminal:
                            tiles = capture(changed=changed)
                            for item in active:
                                if substep == stride or item["done"]:
                                    record_frame(item, tiles, terminal=item["done"])
                    if current_step >= max_steps:
                        break
                chunk += 1
                status.update(status="evaluating", completed_episodes=len(completed),
                    active_episode_ids=[item["id"] for item in batch if not item["done"]],
                    batch_physics_index=current_step, chunk_index=chunk)
                write_json(args.output / "status.json", status)
                if chunk % 5 == 0:
                    print(f"POLICY_PROGRESS batch={offset} sim_seconds={current_step * dt:.3f} chunk={chunk}", flush=True)
        finally:
            for item in batch:
                for video in item["videos"]:
                    video.close()
        for item in batch:
            physics = {key: np.asarray([row[key] for row in item["physics"]]) for key in item["physics"][0]}
            frames = {key: np.asarray([row[key] for row in item["frames"]]) for key in item["frames"][0]}
            np.savez_compressed(item["directory"] / "physics.npz", **physics)
            np.savez_compressed(item["directory"] / "frames.npz", **frames)
            for name in ("wrist", "global"):
                validate_encoded_video(item["directory"] / f"{name}.mp4", len(frames["state"]), fps)
                final_path = item["directory"] / f"{name}_final.jpg"
                if not final_path.exists():
                    Image.fromarray(tiles[2 * item["eid"] + (name == "global")]).save(final_path)
            presses = [event["floor"] for event in item["events"] if event["type"] == "pressed"]
            episode = dict(episode_id=item["id"], floor=item["floor"], task=item["task"],
                seed_episode_index=item["seed_episode_index"],
                **{key: item[key] for key in LAYOUT_FIELDS}, panel_layout=item["panel_layout"],
                env_offset_m=envs[item["eid"]].offset.tolist(),
                repeat=item["repeat"], success=item["termination"] == "target_pressed",
                task_success=presses == [item["floor"]], termination=item["termination"],
                error=item.get("error"), events=item["events"], unexpected_collisions=unintended[item["eid"]],
                config=cfg, initial_joint_velocity_rad_s=item["initial_joint_velocity_rad_s"],
                num_frames=len(item["frames"]), physics_steps=len(item["physics"]),
                sim_seconds=float(physics["sim_time"][-1]), wall_seconds=time.monotonic() - item["started_wall"],
                inference_requests=len(item["requests"]), executed_actions=len(item["actions"]),
                fps=fps, physics_hz=120, capture_stride=stride, action_chunk=7,
                motion_smoothing=motion_smoothing,
                ik_diagnostic_endpoint="q_target before smoothing; executed_* records actual filtered interval endpoint",
                final_partial_interval=bool(physics["physics_index"][-1] % stride),
                pose_frame="base_link", pose_link="gripper_tcp", quaternion_order="wxyz",
                tcp_offset_link6_m=[0, 0, .1358], robot_base_world_m=envs[item["eid"]].base_position.tolist(),
                controller_gripper_width_m=.008, gripper_is_learned=False,
                min_target_tip_distance_m=float(physics["target_tip_distance_m"].min()),
                final_target_tip_distance_m=float(physics["target_tip_distance_m"][-1]),
                max_joint_tracking_error_rad=float(np.max(np.abs(physics["q_actual"] - physics["q_command"]))),
                initial_home_error_rad=float(np.max(np.abs(physics["q_actual"][0] - home))),
                max_command_position_residual_m=max((row["command_position_residual_m"] for row in item["actions"]), default=0.),
                max_command_rotation_residual_rad=max((row["command_rotation_residual_rad"] for row in item["actions"]), default=0.),
                velocity_limited_actions=sum(bool(row["velocity_saturated_joints"]) for row in item["actions"]),
                recorded_actions_used=False, target_planner_used=False,
                source_files={name: file_identity(item["directory"] / name) for name in
                              ("physics.npz", "frames.npz", "wrist.mp4", "global.mp4", "requests.jsonl")})
            write_json(item["directory"] / "metadata.json", episode)
            completed.append({key: episode[key] for key in ("episode_id", "floor", "task", "success", "task_success",
                "termination", "sim_seconds", "inference_requests", "min_target_tip_distance_m",
                "max_command_position_residual_m", "velocity_limited_actions", "repeat", *LAYOUT_FIELDS)})
            print("POLICY_RESULT " + json.dumps(completed[-1]), flush=True)
        write_json(args.output / "report.json", dict(complete=False, episodes=completed))
    passed = sum(row["success"] for row in completed)
    report = dict(complete=True, total_episodes=len(completed), passed_episodes=passed,
        success_rate=passed / len(completed), task_successes=sum(row["task_success"] for row in completed),
        fps=fps, action_chunk_size=7, max_sim_seconds=args.max_seconds, episodes=completed,
        motion_smoothing=motion_smoothing, panel_layout_mode=args.panel_layouts, panel_layouts=layouts,
        finished_at=datetime.now(timezone.utc).isoformat())
    write_json(args.output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18765/predict")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/piper_elevator_lerobot_press_30hz")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "outputs/edge30_source/scene.usda")
    parser.add_argument("--config", type=Path, default=ROOT / "outputs/edge30_source/config.json")
    parser.add_argument("--asset-bundle", type=Path,
                        help="Verified asset bundle for relocating a frozen snapshot across machines; original scene identity is retained")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--floors", default=",".join(str(floor) for floor in range(24, 36)))
    parser.add_argument("--episodes-per-floor", type=int, default=1)
    parser.add_argument("--panel-layouts", choices=("fixed", "center_corners"), default="fixed",
                        help="Fixed scene, or each floor at the recorded XY range center and four corners; repeats apply per position")
    parser.add_argument("--num-envs", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=15.)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--timeout", type=float, default=120.)
    parser.add_argument("--expected-checkpoint-sha256", default="6793045b0b00e863737fe4c0742fde688472d415ab4d8a788ded6254ebe1d1f4")
    parser.add_argument("--expected-checkpoint-step", type=int, default=2000)
    parser.add_argument("--smoothing-window", type=int, choices=(1, 3, 5, 7, 9, 11), default=DEFAULT_SMOOTHING_WINDOW,
                        help="Causal mean window at 120 Hz after joint interpolation; 1 disables filtering")
    args = parser.parse_args()
    args.floors = [int(value) for value in args.floors.split(",")]
    if (not args.floors or len(args.floors) != len(set(args.floors)) or any(floor not in range(24, 36) for floor in args.floors)
            or min(args.num_envs, args.episodes_per_floor) < 1 or not 0 < args.max_seconds <= 120
            or abs(args.max_seconds * 120 - round(args.max_seconds * 120)) > 1e-7
            or args.expected_checkpoint_step < 1):
        parser.error("Require distinct floors24..35, positive counts, duration on a120Hz tick")
    for name in ("dataset", "snapshot", "config", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.asset_bundle is not None:
        args.asset_bundle = args.asset_bundle.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text())
    collection = json.loads((args.dataset / "meta/collection_metadata.json").read_text())
    if cfg != collection["config"] or file_identity(args.snapshot)["sha256"] != collection["scene_sha256"]:
        raise ValueError("Evaluation config/snapshot differs from the recorded training scene")
    layouts = evaluation_panel_layouts(cfg, args.panel_layouts)
    schedule = evaluation_schedule(args.floors, args.episodes_per_floor, layouts)
    health_url = args.endpoint.rsplit("/", 1)[0] + "/health"
    with urlopen(health_url, timeout=args.timeout) as response:
        health = json.load(response)
    if (health.get("model_sha256") != args.expected_checkpoint_sha256
            or health.get("checkpoint_step") != args.expected_checkpoint_step or not health.get("checkpoint_verified")
            or health.get("status") != "ready" or health.get("fps") != 30
            or health.get("action_horizon") != 7 or health.get("camera_order") != ["global", "wrist"]):
        raise ValueError("Policy service is not serving the requested final checkpoint SHA256")
    write_json(args.output / "policy_service.json", health)
    import importlib.metadata
    runtime_versions = {name: importlib.metadata.version(name) for name in
                        ("isaacsim", "torch", "numpy", "scipy", "Pillow")}
    source_names = ["scripts/eval_policy.py", "src/pressb/policy_eval.py", "src/pressb/replay_control.py",
                    "src/pressb/dataset_scene.py", "src/pressb/motion_smoothing.py", "src/pressb/policy_layouts.py",
                    "scripts/collect_dataset.py"]
    if args.asset_bundle is not None:
        source_names.append("src/pressb/scene_portability.py")
    write_json(args.output / "eval_manifest.json", dict(
        started_at=datetime.now(timezone.utc).isoformat(), arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        config=cfg, scene_sha256=collection["scene_sha256"], collection_fingerprint=collection["collection_fingerprint"],
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_checkpoint_step=args.expected_checkpoint_step,
        motion_smoothing=smoothing_settings(args.smoothing_window),
        panel_layout_mode=args.panel_layouts, panel_layouts=layouts, episode_schedule=schedule,
        panel_layout_policy="Predeclared fixed scene positions; no expert planning or target coordinates enter the policy or pose controller",
        model_input="live global RGB, live wrist RGB, measured base_link gripper TCP pose, task sentence",
        policy_control="7 absolute targets per request; bounded single-seed pose IK; four120Hz joint interpolation steps per target followed by causal joint mean filter",
        controller_projection="raw XYZ/quaternion targets unchanged; bounded IK residuals and actual filtered execution residuals recorded separately",
        runtime_versions=runtime_versions,
        renderer_settings=dict(anti_aliasing=DATASET_ANTI_ALIASING,
            light_settle_captures=LIGHT_SETTLE_CAPTURES, light_settle_subframes=LIGHT_SETTLE_SUBFRAMES,
            history_controls=RENDER_HISTORY_CONTROLS, exposure_controls=RENDER_EXPOSURE_CONTROLS),
        robot_urdf=file_identity(ROOT / cfg["robot_urdf"]),
        inference_clock="physics frozen until response; simulation-time evaluation does not measure real-time control latency",
        inference_seed_rule=("base_seed + (repeat * 12 + floor - 24) * 10000 + chunk_index" if args.panel_layouts == "fixed"
                             else "base_seed + ((repeat * num_panel_layouts + panel_layout_index) * 12 + floor - 24) * 10000 + chunk_index"),
        success="target button physical travel>=threshold AND stylus force>0.02N, no wrong press or unexpected collision; no retreat",
        video_timing="30Hz CFR with initial frame; terminal frame can occur before next regular sample, see physics_index",
        recorded_actions_used=False, target_planner_used=False,
        sources={name: file_identity(ROOT / name) for name in source_names}))
    status = dict(status="initializing", started_at=datetime.now(timezone.utc).isoformat(), completed_episodes=0)
    app, exit_code = None, 1
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    try:
        write_json(args.output / "status.json", status)
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "create_new_stage": False, "width": 640, "height": 480,
            "active_gpu": args.gpu, "physics_gpu": args.gpu, "multi_gpu": False,
            "renderer": "RayTracedLighting", "anti_aliasing": DATASET_ANTI_ALIASING, "fast_shutdown": True,
            "extra_args": ["--/app/asyncRendering=false", "--/rtx/ecoMode/enabled=false",
                "--/plugins/carb.tasking.plugin/threadCount=8", "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8",
                "--/persistent/physics/numThreads=4", "--/validate/p2p/enabled=false", "--/validate/iommu/enabled=false"]})
        report = run(app, args, cfg, status)
        status.update(status="complete", completed_episodes=report["total_episodes"],
                      passed_episodes=report["passed_episodes"], finished_at=datetime.now(timezone.utc).isoformat())
        # A completed evaluation with policy failures is a valid result.
        exit_code = 0
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=datetime.now(timezone.utc).isoformat())
        traceback.print_exc()
    finally:
        write_json(args.output / "status.json", status)
        if app is not None:
            app.app.post_quit(exit_code)
            app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
