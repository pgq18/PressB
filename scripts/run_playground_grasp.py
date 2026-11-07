#!/usr/bin/env python3
"""Physics-only green-cube grasp, lift and return in the playground twin.

Never imports a real robot driver or accesses CAN. Objects move only through
PhysX contact and gravity. Plans are SI joint targets, not hardware execution
permissions. The default empty playground is preserved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pressb.robots.piper.kinematics import PiperKinematics
from pressb.runtime.recording import Video, write


def resolve_grasp_heights(cfg, *, grasp_height=None, grasp_height_above_table=None):
    """Resolve a TCP target into the base frame used by URDF IK.

    New plans specify height above the tabletop. An explicitly supplied
    ``grasp_height`` retains the old base-frame meaning, including its 80 mm
    base-frame pregrasp. A negative base-frame Z is valid when the mounting
    plate raises the base above a low tabletop-relative target.
    """
    if grasp_height is not None and grasp_height_above_table is not None:
        raise ValueError("Specify either grasp_height (legacy base Z) or grasp_height_above_table")
    mount = float(cfg.get("robot_base_height_above_table_m", 0.))
    if not math.isfinite(mount) or mount < 0:
        raise ValueError("robot_base_height_above_table_m must be finite and nonnegative")
    legacy = grasp_height is not None
    requested = float(grasp_height if legacy else
                      (.020 if grasp_height_above_table is None else grasp_height_above_table))
    if not math.isfinite(requested):
        raise ValueError("Grasp height must be finite")
    above_table = requested+mount if legacy else requested
    pregrasp_above_table = .080+mount if legacy else .080
    if not 0 < above_table < pregrasp_above_table:
        raise ValueError("Grasp TCP must be above the tabletop and below the pregrasp TCP")
    return dict(input_reference="legacy_robot_base_z" if legacy else "tabletop_z",
        requested_height_m=requested, robot_base_height_above_table_m=mount,
        table_top_base_z_m=-mount, grasp_height_above_table_m=above_table,
        grasp_height_base_m=requested if legacy else requested-mount,
        pregrasp_height_above_table_m=pregrasp_above_table,
        pregrasp_height_base_m=.080 if legacy else .080-mount)


def make_plan(cfg, *, grasp_height=None, grasp_height_above_table=None, close_width=.026):
    heights = resolve_grasp_heights(cfg, grasp_height=grasp_height,
                                   grasp_height_above_table=grasp_height_above_table)
    grasp_height = heights["grasp_height_base_m"]
    pregrasp_height = heights["pregrasp_height_base_m"]
    close_width = float(close_width)
    if not math.isfinite(close_width) or not .020 <= close_width <= .028:
        raise ValueError("Close width must be finite and 20–28 mm")
    kin = PiperKinematics(ROOT / cfg["robot_urdf"], tip_offset=.1358)
    initial = np.array(cfg["initial_state_joint_rad_m"], dtype=float)
    obj = next(value for value in cfg["objects"] if value["name"] == "GreenCube")
    x, y = obj["center_xy"]
    yaw = obj.get("yaw_deg", 0.) + 180.
    rotation = Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ np.diag([1., -1., -1.])
    q = initial[:6].copy()
    records = []

    def add(name, state, hold=1.):
        records.append(dict(name=name, action=np.asarray(state).tolist(), hold_s=hold,
                            tcp_base_m=kin.fk(np.asarray(state)[:6])[:3, 3].tolist()))

    add("open", np.r_[q, .060])
    q = kin.solve(np.array([x, y, pregrasp_height]), q, rotation)
    add("pregrasp", np.r_[q, .060])
    # Cartesian samples avoid a bowed joint-space descent near the tabletop.
    for z in np.linspace(pregrasp_height, grasp_height, 13)[1:]:
        q = kin.solve(np.array([x, y, z]), q, rotation)
        add("descend" if z != grasp_height else "grasp_open", np.r_[q, .060], 0.)
    records[-1]["hold_s"] = 1.
    add("close", np.r_[q, close_width], 1.5)
    for z in np.linspace(grasp_height, grasp_height+.050, 11)[1:]:
        q = kin.solve(np.array([x, y, z]), q, rotation)
        add("lift", np.r_[q, close_width], 0.)
    records[-1]["name"], records[-1]["hold_s"] = "lift_hold", 2.
    for z in np.linspace(grasp_height+.050, grasp_height, 11)[1:]:
        q = kin.solve(np.array([x, y, z]), q, rotation)
        add("lower", np.r_[q, close_width], 0.)
    records[-1]["hold_s"] = 1.
    add("release", np.r_[q, .060], 1.5)
    for z in np.linspace(grasp_height, pregrasp_height, 13)[1:]:
        q = kin.solve(np.array([x, y, z]), q, rotation)
        add("retreat", np.r_[q, .060], 0.)
    records[-1]["hold_s"] = 2.
    previous = initial.copy()
    speed = np.r_[np.full(6, np.deg2rad(5.)), .010]
    commands, phases = [], []
    fps = cfg["fps"]
    for row in records:
        destination = np.array(row["action"])
        steps = max(1, math.ceil(float(np.max(np.abs(destination-previous)/speed))*fps))
        row["start_frame"] = len(commands)
        for i in range(1, steps+1):
            commands.append((previous+(destination-previous)*(i/steps)).tolist())
            phases.append(row["name"])
        row["arrival_frame"] = len(commands)-1
        for _ in range(round(row["hold_s"]*fps)):
            commands.append(destination.tolist())
            phases.append(row["name"])
        row["end_frame"] = len(commands)-1
        previous = destination
    commands = np.array(commands)
    fk = np.array([kin.fk(state[:6]) for state in commands])
    if np.any(commands[:, :6] < kin.lower) or np.any(commands[:, :6] > kin.upper):
        raise ValueError("Plan violates URDF joint limits")
    if fk[:, 2, 3].min() < grasp_height-1e-5:
        raise ValueError("Interpolated TCP descends below the reviewed grasp height")
    return dict(schema_version=1, hardware_access=False, units=["rad"]*6+["m"],
        expected_initial_state=initial.tolist(), fps=fps, actions=commands.tolist(), phases=phases,
        waypoints=records, object=copy.deepcopy(obj), tip_offset_m=.1358,
        # Retain the historical field's base-frame meaning for downstream readers.
        grasp_height_m=grasp_height, **heights,
        grasp_geometry={"tcp_definition": "link6 origin plus 0.1358 m along link6 +Z",
            "object_height_m": obj["size"][2],
            "object_spawn_clearance_m": obj.get("clearance", .0005),
            "nominal_settled_object_top_above_table_m": obj["size"][2],
            "nominal_tcp_depth_below_settled_object_top_m": obj["size"][2]-heights["grasp_height_above_table_m"],
            "interpretation": "TCP height is not a finger-body clearance or measured contact depth; "
                              "spawn clearance settles under gravity and is not added to the TCP target."},
        lift_distance_m=.050, open_width_m=.060, close_width_m=close_width,
        max_joint_speed_rad_s=float(speed[0]), max_gripper_speed_m_s=float(speed[6]),
        duration_s=len(commands)/fps, min_planned_tcp_z_m=float(fk[:, 2, 3].min()),
        source={"kind": "URDF IK with Cartesian descent/lift", "height_reference": copy.deepcopy(heights),
                "configuration": copy.deepcopy(cfg)})


def validate_input_plan(plan, cfg, *, allow_incomplete=False):
    """Validate physics targets without importing Isaac or real-hardware code."""
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ValueError("Input plan must use schema_version=1")
    actions = np.asarray(plan.get("actions"), dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("Input actions must be a finite nonempty N x 7 rad/m array")
    initial = np.asarray(plan.get("expected_initial_state"), dtype=float)
    if initial.shape != (7,) or not np.isfinite(initial).all():
        raise ValueError("Input expected_initial_state must contain seven finite rad/m values")
    kin = PiperKinematics(ROOT/cfg["robot_urdf"], tip_offset=.1358)
    lower = np.r_[np.maximum(kin.lower, np.deg2rad([-150, 0, -170, -100, -70, -180])), 0.]
    upper = np.r_[np.minimum(kin.upper, np.deg2rad([150, 180, 0, 100, 70, 180])), cfg["gripper_max_width_m"]]
    if np.any(actions < lower-1e-8) or np.any(actions > upper+1e-8) or np.any(initial < lower-1e-8) or np.any(initial > upper+1e-8):
        raise ValueError("Input plan exceeds the intersection of URDF and hardware command bounds")
    phases = plan.get("phases")
    if not isinstance(phases, list) or len(phases) != len(actions) or any(not isinstance(p, str) or not p for p in phases):
        raise ValueError("Input plan requires one nonempty phase label per command")
    if plan.get("fps") != cfg["fps"]:
        raise ValueError("Input plan FPS must match the scene control FPS")
    if "lift_hold" not in phases and not allow_incomplete:
        raise ValueError("Input plan must include a lift_hold phase for physical outcome auditing")
    return copy.deepcopy(plan)


def retime_applied_commands(plan, logs, offsets, *, start_unix=None, end_unix=None):
    """Zero-order hold actual applied targets on a common absolute time axis.

    This function never reads ``observed``. The physics initial condition stays
    the reviewed plan's initial state. If video starts after the first command,
    the omitted prefix is still physically simulated; no dynamic pose is set.
    Between logs the last logged applied target is held. The runner's cleanup
    hold command is not logged, so that narrow distinction is explicitly noted.
    """
    if not logs or len(logs) != len(offsets):
        raise ValueError("Supply one action offset for every real replay log")
    events, sources, last_time = [], [], -float("inf")
    for path, offset in zip(logs, offsets):
        path = Path(path)
        if type(offset) is not int or offset < 0:
            raise ValueError("Log action offsets must be nonnegative integers")
        first_event = len(events)
        for number, line in enumerate(path.read_text().splitlines(), 1):
            row = json.loads(line)
            stamp = row.get("unix_s")
            action_index = row.get("action_index")
            command = np.asarray(row.get("applied"), dtype=float)
            if isinstance(stamp, bool) or not isinstance(stamp, (float, int)) or not math.isfinite(stamp) or stamp <= last_time:
                raise ValueError(f"Non-increasing or invalid applied-command time at {path}:{number}")
            if type(action_index) is not int or not 0 <= action_index+offset < len(plan["actions"]):
                raise ValueError(f"Invalid original action index at {path}:{number}")
            if command.shape != (7,) or not np.isfinite(command).all():
                raise ValueError(f"No finite seven-dimensional applied command at {path}:{number}")
            events.append((float(stamp), command, plan["phases"][action_index+offset], len(sources), number))
            last_time = float(stamp)
        if len(events) == first_event:
            raise ValueError(f"Empty replay log: {path}")
        sources.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            original_action_offset=offset, rows=len(events)-first_event,
                            first_unix_s=events[first_event][0], last_unix_s=events[-1][0]))
    video_start = events[0][0] if start_unix is None else float(start_unix)
    end = events[-1][0]+1/plan["fps"] if end_unix is None else float(end_unix)
    if not math.isfinite(video_start) or not math.isfinite(end) or end <= video_start or end < events[0][0]:
        raise ValueError("Requested wall-time interval is empty or invalid")
    origin = min(video_start, events[0][0])
    fps = plan["fps"]
    frames = math.ceil((end-origin)*fps)
    if frames > fps*3600:
        raise ValueError("Refusing a replay longer than one hour")
    stamps = np.array([event[0] for event in events])
    actions, phases, provenance = [], [], []
    for frame in range(frames):
        stamp = origin+frame/fps
        index = int(np.searchsorted(stamps, stamp, side="right"))-1
        if index < 0:
            command, phase = plan["expected_initial_state"], "initial_hold"
            provenance.append(None)
        else:
            _, command, phase, source_index, line = events[index]
            provenance.append(dict(source_index=source_index, line=line, applied_unix_s=float(stamps[index])))
        actions.append(np.asarray(command).tolist())
        phases.append(phase)
    waypoints, run_start = [], 0
    for frame in range(len(phases)):
        if frame == len(phases)-1 or phases[frame+1] != phases[frame]:
            waypoints.append(dict(name=phases[frame], action=actions[frame], start_frame=run_start,
                                  arrival_frame=frame, end_frame=frame, hold_s=0.))
            run_start = frame+1
    result = copy.deepcopy(plan)
    result.update(actions=actions, phases=phases, waypoints=waypoints, duration_s=frames/fps,
        timing=dict(kind="physics rerun of actual applied commands", simulation_start_unix_s=origin,
                    requested_video_start_unix_s=video_start, requested_end_unix_s=end,
                    video_start_frame=max(0, math.ceil((video_start-origin)*fps-1e-6)),
                    sampling="zero-order hold of latest applied target at start of each control interval",
                    timestamp_precision="unix_s is recorded after command send and feedback read; no hardware clock synchronization",
                    gap_behavior="last applied target retained across logs; runner cleanup hold target is not recorded",
                    sources=sources), command_provenance=provenance)
    result["source"] = {"kind": "real applied commands with original wall timing", "original_plan_source": plan["source"]}
    return result


def observation_audit_samples(plan):
    """Read real feedback only for a non-driving comparison with the physics state."""
    if "timing" not in plan:
        return [None]*len(plan["actions"])
    sources = [[json.loads(line) for line in Path(source["path"]).read_text().splitlines()]
               for source in plan["timing"]["sources"]]
    result = []
    for item in plan["command_provenance"]:
        if item is None:
            result.append(None)
            continue
        row = sources[item["source_index"]][item["line"]-1]
        observed = np.asarray(row.get("observed"), dtype=float)
        if observed.shape != (7,) or not np.isfinite(observed).all():
            raise ValueError("Real observation audit requires finite seven-dimensional feedback")
        result.append(observed)
    return result


def simulate(args, cfg, plan):
    from pxr import PhysicsSchemaTools
    import omni.physx
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
    from pressb.robots.piper.isaac import build_piper_runtime
    from pressb.runtime.workbench import build_tripod
    from pressb.sensors.camera import make_camera, set_intrinsics
    from pressb.tasks.playground.scene import build_scene
    from PIL import Image

    runtime = build_piper_runtime(cfg, build_scene, asset_root=ROOT,
        camera_hardware_builder=build_tripod if cfg.get("camera_tripod", True) else None)
    world, robot, stage = runtime.world, runtime.robot, runtime.stage
    object_path = runtime.scene["objects"]["GreenCube"]["path"]
    cube = SingleRigidPrim(prim_path=object_path, name="green_cube_measurement")
    cube.initialize()
    overview = cfg["overview_camera"]
    path = make_camera(stage, runtime.scene["root_path"]+"/OverviewCamera", overview, runtime.base)
    camera = Camera(prim_path=path, name="overview", resolution=tuple(overview["resolution"]), frequency=-1)
    camera.initialize()
    runtime.cameras["overview"] = camera
    runtime.camera_specs["overview"] = (path, overview["resolution"])
    for name in runtime.cameras:
        set_intrinsics(stage.GetPrimAtPath(runtime.camera_specs[name][0]), cfg[name+"_camera"])
    cumulative, active, unintended = {"link7": 0, "link8": 0}, {}, []
    physics_tick = 0

    def contact(headers, data):
        for h in headers:
            if not h.num_contact_data:
                continue
            actors = [str(PhysicsSchemaTools.intToSdfPath(value)) for value in (h.actor0, h.actor1)]
            pair = tuple(sorted(actors))
            active[pair] = active.get(pair, 0)+1
            if object_path in actors:
                for finger in cumulative:
                    if "/World/Piper/"+finger in actors:
                        cumulative[finger] += 1
            robot_actors = [actor for actor in actors if actor.startswith("/World/Piper/")]
            if robot_actors and object_path not in actors:
                # Adjacent links can have solver contacts without scene impact;
                # retain all pairs for audit and flag external dynamic contact.
                external = any(not actor.startswith("/World/Piper/") for actor in actors)
                if external and not any(actor.endswith("/base_link") for actor in robot_actors):
                    unintended.append(dict(physics_tick=physics_tick, actors=actors))

    subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contact)
    for _ in range(120):
        world.step(render=False)
        physics_tick += 1
    for _ in range(runtime.warmup_frames):
        world.render()

    def images(*, flush=False):
        world.physics_sim_view.update_articulations_kinematic()
        omni.physx.get_physx_interface().update_transformations(False, True, True)
        for _ in range(16 if flush else runtime.subframes):
            world.render()
        return {name: np.ascontiguousarray(cam.get_rgba()[..., :3]) for name, cam in runtime.cameras.items()}

    for name, rgb in images(flush=True).items():
        Image.fromarray(rgb).save(args.output/("initial_"+name+".jpg"))
    start_pos = np.array(cube.get_world_pose()[0])-runtime.base
    pipes = {name: Video(args.output/(name+".mp4"), size, runtime.fps)
             for name, (_, size) in runtime.camera_specs.items()} if args.video else {}
    rows, event_last = [], None
    video_start_frame = plan.get("timing", {}).get("video_start_frame", 0)
    audit_samples = observation_audit_samples(plan)
    audit_kin = PiperKinematics(ROOT/cfg["robot_urdf"], tip_offset=.1358)
    try:
        with (args.output/"physics.jsonl").open("w") as log:
            for frame, (command, phase) in enumerate(zip(plan["actions"], plan["phases"])):
                active.clear()
                runtime.controller.apply_action(ArticulationAction(joint_positions=runtime.full_command(command)))
                for _ in range(runtime.stride):
                    world.step(render=False)
                    physics_tick += 1
                values = robot.get_joint_positions()[runtime.indices]
                state = np.r_[values[:6], values[6]-values[7]]
                pos, quat = cube.get_world_pose()
                pos = np.array(pos)-runtime.base
                row = dict(frame=frame, elapsed_s=(frame+1)/runtime.fps, phase=phase,
                    commanded=command, observed=state.tolist(), object_base_m=pos.tolist(),
                    object_quat_wxyz=np.asarray(quat).tolist(), object_velocity=np.asarray(cube.get_linear_velocity()).tolist(),
                    cumulative_finger_contacts=copy.deepcopy(cumulative),
                    contacts=[dict(actors=list(pair), events=count) for pair, count in active.items()])
                if "timing" in plan:
                    row["wall_unix_s"] = plan["timing"]["simulation_start_unix_s"]+(frame+1)/runtime.fps
                    row["command_source"] = plan["command_provenance"][frame]
                if audit_samples[frame] is not None:
                    real_observed = audit_samples[frame]
                    row["non_driving_real_observation_audit"] = dict(
                        observed=real_observed.tolist(), sim_minus_real_joint_rad_m=(state-real_observed).tolist(),
                        sim_minus_real_fk_tcp_m=(audit_kin.fk(state[:6])[:3, 3]-audit_kin.fk(real_observed[:6])[:3, 3]).tolist())
                log.write(json.dumps(row)+"\n")
                rows.append(row)
                if args.video and frame >= video_start_frame:
                    # A cropped video may follow a long non-rendered physics
                    # prefix. Flush buffered images at the first saved frame.
                    for name, rgb in images(flush=frame == video_start_frame).items():
                        pipes[name].add(rgb)
                if phase != event_last:
                    print(json.dumps(dict(event="phase", phase=phase, frame=frame,
                        object_base_m=pos.tolist(), finger_contacts=cumulative)), flush=True)
                    event_last = phase
                if frame % 300 == 0:
                    log.flush()
                if phase == "lift_hold" and frame == next(v["end_frame"] for v in plan["waypoints"] if v["name"] == "lift_hold"):
                    for name, rgb in images(flush=True).items():
                        Image.fromarray(rgb).save(args.output/("lift_"+name+".jpg"))
    finally:
        for pipe in pipes.values():
            pipe.close()
    for name, rgb in images(flush=True).items():
        Image.fromarray(rgb).save(args.output/("final_"+name+".jpg"))
    held = [row for row in rows if row["phase"] == "lift_hold"][-round(runtime.fps*1.5):]
    held_height = [row["object_base_m"][2]-start_pos[2] for row in held]
    final = rows[-1]
    contacts_at_hold = bool(held) and all(held[-1]["cumulative_finger_contacts"].values())
    def both_fingers(row):
        touched = {actor.rsplit("/", 1)[-1] for entry in row["contacts"]
                   if object_path in entry["actors"] for actor in entry["actors"]
                   if actor.startswith("/World/Piper/")}
        return {"link7", "link8"} <= touched
    bilateral_hold_fraction = sum(both_fingers(row) for row in held)/len(held) if held else None
    position_excursion = angle_excursion = None
    if held:
        held_positions = np.array([row["object_base_m"] for row in held])
        held_rotations = Rotation.from_quat(np.array([row["object_quat_wxyz"] for row in held])[:, [1, 2, 3, 0]])
        position_excursion = float(np.max(np.linalg.norm(held_positions-held_positions[0], axis=1)))
        angle_excursion = float(np.max((held_rotations[0].inv()*held_rotations).magnitude()))
    # PhysX contact-solver velocity can retain a nonzero correction while the
    # resulting poses are stationary. Evaluate physical rest from the measured
    # 1.5-second pose history; preserve raw velocities for independent review.
    stable = bool(held) and position_excursion < .001 and angle_excursion < np.deg2rad(1.)
    return_distance = float(np.linalg.norm(np.array(final["object_base_m"])-start_pos))
    complete = bool(held) and "release" in plan["phases"] and "retreat" in plan["phases"]
    success = bool(min(held_height) >= .045 and stable and contacts_at_hold and bilateral_hold_fraction == 1.
                   and return_distance < .010 and not unintended) if complete else None
    contact_rows = [row for row in rows if any(object_path in entry["actors"] and
        any(actor.startswith("/World/Piper/") for actor in entry["actors"]) for entry in row["contacts"])]
    phase_summary = {}
    for phase in dict.fromkeys(plan["phases"]):
        selected = [row for row in rows if row["phase"] == phase]
        touches = [row for row in contact_rows if row["phase"] == phase]
        phase_summary[phase] = dict(frames=len(selected), cube_contact_frames=len(touches),
            first_contact_elapsed_s=touches[0]["elapsed_s"] if touches else None,
            max_cube_xy_displacement_m=max(float(np.linalg.norm(np.array(row["object_base_m"][:2])-start_pos[:2])) for row in selected),
            min_gripper_width_m=min(row["observed"][6] for row in selected))
    audited = [row["non_driving_real_observation_audit"] for row in rows if "non_driving_real_observation_audit" in row]
    audit_summary = None
    if audited:
        audit_summary = dict(drives_simulation=False, sampled_frames=len(audited),
            max_abs_joint_error_rad_m=np.max(np.abs([row["sim_minus_real_joint_rad_m"] for row in audited]), axis=0).tolist(),
            max_fk_tcp_error_m=max(float(np.linalg.norm(row["sim_minus_real_fk_tcp_m"])) for row in audited))
    report = dict(success=success, complete_grasp_return_experiment=complete,
        outcome_status="evaluated_full_grasp_return" if complete else "partial_execution_no_full_grasp_claim",
        simulation_only=True, hardware_access=False,
        attachments_or_object_teleportation=False, initial_object_base_m=start_pos.tolist(),
        min_held_lift_m=min(held_height) if held else None, max_lift_m=max(row["object_base_m"][2]-start_pos[2] for row in rows),
        held_stable=bool(stable), bilateral_contact_history=bool(contacts_at_hold),
        held_bilateral_contact_fraction=bilateral_hold_fraction,
        stable_window_s=len(held)/runtime.fps, held_position_excursion_m=position_excursion,
        held_rotation_excursion_rad=angle_excursion,
        held_max_reported_solver_velocity_m_s=max(float(np.linalg.norm(row["object_velocity"])) for row in held) if held else None,
        cumulative_finger_contacts=cumulative, return_distance_m=return_distance,
        final_object_base_m=final["object_base_m"], unintended_external_robot_contacts=unintended,
        physics_ticks=physics_tick, fps=runtime.fps, frames=len(rows), duration_s=len(rows)/runtime.fps,
        timing=plan.get("timing"), video_start_frame=video_start_frame,
        video_frame_count=max(0, len(rows)-video_start_frame) if args.video else 0,
        phase_contact_summary=phase_summary, non_driving_real_observation_audit=audit_summary,
        first_robot_cube_contact={key: contact_rows[0].get(key) for key in ("frame", "elapsed_s", "wall_unix_s", "phase", "observed", "object_base_m", "contacts")} if contact_rows else None,
        video_paths={name: str(pipe.path) for name, pipe in pipes.items()},
        plan_sha256=hashlib.sha256((args.output/"plan.json").read_bytes()).hexdigest())
    write(args.output/"summary.json", report)
    print(json.dumps(dict(event="complete", **report)), flush=True)
    return success if complete else args.allow_incomplete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/piper_playground.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    heights = parser.add_mutually_exclusive_group()
    heights.add_argument("--grasp-height-above-table", type=float,
                         help="TCP height above the tabletop in metres (default: 0.020); mounting height is subtracted for IK")
    heights.add_argument("--grasp-height", "--grasp-height-base", type=float,
                         help="Legacy explicit TCP Z in the robot base frame; keeps the old 0.080 m base-frame pregrasp")
    parser.add_argument("--close-width", type=float, default=.026)
    parser.add_argument("--plan-input", type=Path, help="Existing SI physics plan; no real hardware is accessed")
    parser.add_argument("--retime-log", type=Path, action="append", default=[],
                        help="Chronological real replay JSONL; requires --plan-input; may repeat")
    parser.add_argument("--log-action-offset", type=int, action="append", default=[],
                        help="Original dense-plan action index of the first action in each log")
    parser.add_argument("--start-unix", type=float, help="First output video time; earlier command prefix still runs in physics")
    parser.add_argument("--end-unix", type=float, help="Final absolute wall time to simulate")
    parser.add_argument("--allow-incomplete", action="store_true", help="Audit a stopped trajectory without claiming a full grasp/lift outcome")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    cfg["objects"] = copy.deepcopy(cfg["calibration_objects"])
    cfg["initial_joint_positions_rad"] = cfg["initial_state_joint_rad_m"][:6]
    cfg["initial_gripper_width_m"] = cfg["initial_state_joint_rad_m"][6]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.retime_log and not args.plan_input:
        parser.error("--retime-log requires --plan-input")
    if not args.retime_log and (args.log_action_offset or args.start_unix is not None or args.end_unix is not None):
        parser.error("Wall-time options require --retime-log")
    if args.plan_input and (args.grasp_height is not None or args.grasp_height_above_table is not None):
        parser.error("Height options apply only to new plans; --plan-input preserves recorded actions")
    plan = (validate_input_plan(json.loads(args.plan_input.read_text()), cfg, allow_incomplete=args.allow_incomplete) if args.plan_input else
            make_plan(cfg, grasp_height=args.grasp_height, grasp_height_above_table=args.grasp_height_above_table,
                      close_width=args.close_width))
    if args.retime_log:
        plan = retime_applied_commands(plan, args.retime_log, args.log_action_offset,
                                      start_unix=args.start_unix, end_unix=args.end_unix)
        plan = validate_input_plan(plan, cfg, allow_incomplete=args.allow_incomplete)
    cfg["initial_state_joint_rad_m"] = plan["expected_initial_state"]
    cfg["initial_joint_positions_rad"] = plan["expected_initial_state"][:6]
    cfg["initial_gripper_width_m"] = plan["expected_initial_state"][6]
    write(args.output/"plan.json", plan)
    write(args.output/"config.json", cfg)
    if args.plan_only:
        print(json.dumps({"event": "plan", "frames": len(plan["actions"]), "duration_s": plan["duration_s"]}))
        return
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, active_gpu=args.gpu, physics_gpu=args.gpu,
        multi_gpu=False, renderer="RayTracedLighting", anti_aliasing=1, fast_shutdown=True,
        extra_args=["--/app/asyncRendering=false", "--/rtx/ecoMode/enabled=false",
                    "--/validate/p2p/enabled=false", "--/validate/iommu/enabled=false",
                    "--/plugins/carb.tasking.plugin/threadCount=8", "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8"]))
    try:
        success = simulate(args, cfg, plan)
    finally:
        app.close()
    if not success:
        raise SystemExit("Physical grasp/lift/return criteria did not pass; inspect summary.json")


if __name__ == "__main__":
    main()
