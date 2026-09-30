#!/usr/bin/env python3
"""PhysX corner checks for XY panel randomization, with calibrated RGB evidence.

This is a validation run, not a training dataset. It executes every floor at
each configured XY corner and saves 120 Hz physical evidence plus selected
camera images. Use the ordinary collector for synchronized 30 Hz episodes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import product
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from collect_dataset import RENDER_EXPOSURE_CONTROLS, RENDER_HISTORY_CONTROLS, write_json


def simulate(app, args, cfg, status):
    import numpy as np
    from PIL import Image
    from pxr import PhysicsSchemaTools
    import carb
    import omni.physx
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.utils.types import ArticulationAction
    from pressb.dataset_scene import create_envs, build_tiled_rgb, set_panel_offset, validate_panel_layout
    from pressb.kinematics import PiperKinematics
    from pressb.planning import make_plan
    from pressb.panel_randomization import validate_panel_layout as validate_plan_layout
    from pressb.scene import set_button_light

    dt = float(cfg["physics_dt"])
    kin = PiperKinematics(ROOT / cfg["robot_urdf"], tip_offset=cfg["tip_offset"])
    world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=1/30,
                  backend="numpy", device="cpu")
    world.get_physics_context().set_physx_update_transformations_settings(
        update_to_usd=True, update_velocities_to_usd=True)
    env = create_envs(world, args.snapshot, 1, cfg=cfg)[0]
    render_product, annotator, split_rgb = build_tiled_rgb([env.wrist_camera_path, env.global_camera_path])
    rep.orchestrator.set_capture_on_play(False)
    world.reset()
    arm = np.array([env.robot.get_dof_index(name) for name in kin.joint_names])
    fingers = np.array([env.robot.get_dof_index(name) for name in ("joint7", "joint8")])
    home = np.asarray(cfg["home_q"])
    command = env.robot.get_joint_positions().copy()
    command[arm], command[fingers] = home, cfg["gripper_joint_positions_m"]
    env.robot.set_joint_positions(command)
    env.robot.set_joint_velocities(np.zeros_like(command))
    controller = env.robot.get_articulation_controller()
    controller.set_gains(kps=np.full(len(command), 12000.), kds=np.full(len(command), 600.))
    controller.apply_action(ArticulationAction(joint_positions=command))
    forces = np.zeros(12)
    unexpected, tick = [], [-1]
    body_map = {info["body_path"]: floor for floor, info in env.button_info.items()}

    def contact(headers, data):
        for header in headers:
            if not header.num_contact_data:
                continue
            actors = [str(PhysicsSchemaTools.intToSdfPath(v)) for v in (header.actor0, header.actor1)]
            colliders = [str(PhysicsSchemaTools.intToSdfPath(v)) for v in (header.collider0, header.collider1)]
            force = sum(float(np.linalg.norm(data[i].impulse))/dt for i in
                        range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data))
            floor = next((body_map[a] for a in actors if a in body_map), None)
            if floor is not None and any(c in env.tool_colliders for c in colliders):
                forces[floor - 24] += force
                continue
            prefix = env.robot_path + "/"
            robot = [a for a in actors if a.startswith(prefix)]
            if not robot or force <= .1:
                continue
            shaft = prefix + "link6/PressStylus"
            pads = {prefix + "link7/collisions", prefix + "link8/collisions"}
            if shaft in colliders and any(c in pads for c in colliders):
                continue
            reason = None
            if floor is not None:
                reason = "non_stylus_button_contact"
            elif len(robot) == 2 and actors[0] != actors[1]:
                reason = "robot_self_contact"
            elif len(robot) == 1 and robot[0].rsplit("/", 1)[-1] not in ("dummy_link", "base_link", "link1"):
                reason = "robot_environment_contact"
            if reason and len(unexpected) < 100:
                unexpected.append(dict(step=tick[0], actors=actors, colliders=colliders, force_n=force, reason=reason))

    subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contact)
    for i in range(90):
        forces.fill(0.)
        world.step(render=False)
        if i % 4 == 0:
            world.render()
    settings = carb.settings.get_settings()
    for key, value in {**RENDER_HISTORY_CONTROLS, **RENDER_EXPOSURE_CONTROLS}.items():
        settings.set(key, value)
    if unexpected:
        raise RuntimeError(f"Warmup collision: {unexpected[:2]}")

    def capture(directory, name):
        before = world.current_time
        omni.usd.get_context().reset_renderer_accumulation()
        for _ in range(4):
            rep.orchestrator.step(delta_time=0., pause_timeline=False, rt_subframes=2)
        for attempt in range(12):
            rep.orchestrator.step(delta_time=0., pause_timeline=False, rt_subframes=16)
            try:
                frames = split_rgb(annotator.get_data())
                break
            except ValueError:
                if attempt == 11:
                    raise
        if abs(world.current_time - before) > 1e-10:
            raise RuntimeError("Validation capture advanced physics")
        for view, frame in zip(("wrist", "global"), frames):
            if frame.shape != (480, 640, 3) or frame.std() < 2:
                raise RuntimeError("Invalid validation camera image")
            Image.fromarray(frame).save(directory / f"{name}_{view}.png")

    randomization = cfg["panel_randomization"]
    layouts = list(product((randomization["min_offset_x_m"], randomization["max_offset_x_m"]),
                           (randomization["min_offset_y_m"], randomization["max_offset_y_m"])))
    summaries = []
    for index, (x, y) in enumerate(layouts):
        directory = args.output / f"corner_{index}"
        directory.mkdir()
        print(f"LAYOUT {index} x={x} y={y}", flush=True)
        layout_cfg = dict(cfg, panel_offset_x_m=x, panel_offset_y_m=y, sequence=list(range(24, 36)))
        planned = validate_plan_layout(kin, layout_cfg, randomization["camera_margin_px"])
        plan = make_plan(kin, layout_cfg)
        set_panel_offset(world, env, y, offset_x_m=x)
        unexpected.clear()
        for _ in range(24):
            forces.fill(0.)
            world.step(render=False)
        actual_layout = validate_panel_layout(world, env)
        capture(directory, "home")
        rows, events, lit = [], [], set()
        max_error, max_lateral = 0., 0.
        for step, target in enumerate(plan.q):
            if not app.is_running():
                raise RuntimeError("Simulator closed during corner validation")
            tick[0] = step
            command[arm] = target
            controller.apply_action(ArticulationAction(joint_positions=command))
            forces.fill(0.)
            world.step(render=False)
            q = env.robot.get_joint_positions()
            positions = np.array([env.buttons[f].get_world_pose()[0] for f in range(24, 36)])
            rest = np.array([env.button_info[f]["body_center"] for f in range(24, 36)])
            travel = positions[:, 0] - rest[:, 0]
            max_lateral = max(max_lateral, float(np.max(np.abs(positions[:, 1:] - rest[:, 1:]))))
            max_error = max(max_error, float(np.max(np.abs(q[arm] - target))))
            if not np.isfinite(np.r_[q, positions.ravel(), forces]).all():
                raise RuntimeError("Nonfinite corner physics")
            image_floor = None
            for floor in range(24, 36):
                j = floor - 24
                on = floor not in lit and travel[j] >= cfg["press_threshold"] and forces[j] > .02
                off = floor in lit and travel[j] <= cfg["release_threshold"]
                if on or off:
                    lit.add(floor) if on else lit.remove(floor)
                    set_button_light(world.stage, env.button_info[floor], on)
                    events.append(dict(type="pressed" if on else "released", floor=floor, step=step,
                                       travel_m=float(travel[j]), force_n=float(forces[j])))
                    if on and floor in (24, 29, 30, 35):
                        image_floor = floor
            rows.append((q[arm].copy(), target.copy(), q[fingers].copy(), travel, forces.copy(),
                         np.array([int(f in lit) for f in range(24, 36)], dtype=np.uint8)))
            if image_floor is not None:
                capture(directory, f"pressed_{image_floor}")
            if step % 2000 == 0:
                print(f"PHYSICS corner={index} step={step}/{len(plan.q)}", flush=True)
        np.savez_compressed(directory / "physics.npz",
            **{name: np.asarray([row[j] for row in rows]) for j, name in enumerate(
                ("q_actual", "q_command", "gripper_actual", "button_travel", "contact_force", "lights"))})
        expected = [(kind, floor) for floor in range(24, 36) for kind in ("pressed", "released")]
        home_error = float(np.max(np.abs(rows[-1][0] - home)))
        success = ([(e["type"], e["floor"]) for e in events] == expected and not lit and not unexpected
                   and max_error < .15 and home_error < cfg["home_tolerance_rad"] and max_lateral < .0002)
        result = dict(success=success, corner=index, panel_offset_x_m=x, panel_offset_y_m=y,
                      planned=planned, panel_layout=actual_layout, events=events, unexpected_collisions=list(unexpected),
                      physics_steps=len(rows), max_joint_error_rad=max_error,
                      max_button_lateral_error_m=max_lateral, final_home_error_rad=home_error)
        write_json(directory / "report.json", result)
        summaries.append(result)
        status.update(status="checking_corners", completed_corners=len(summaries))
        write_json(args.output / "status.json", status)
        if not success:
            raise RuntimeError(f"Corner {index} failed physical validation: {directory}")
    write_json(args.output / "report.json", dict(success=True, config=cfg, corners=summaries,
        physics_presses=48, capture_kind="selected still frames, not a dataset or continuous video",
        finished_at=datetime.now(timezone.utc).isoformat()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dataset_panel_randomized.json")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "outputs/edge_feedback/scene.usda")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/panel_randomization/corners")
    parser.add_argument("--gpu", type=int, default=4)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text())
    status = dict(status="starting", started_at=datetime.now(timezone.utc).isoformat(), completed_corners=0)
    app, code = None, 1
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    try:
        write_json(args.output / "status.json", status)
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "create_new_stage": False, "width": 640, "height": 480,
            "active_gpu": args.gpu, "physics_gpu": args.gpu, "multi_gpu": False,
            "renderer": "RayTracedLighting", "anti_aliasing": 2, "fast_shutdown": True,
            "extra_args": ["--/app/asyncRendering=false", "--/rtx/ecoMode/enabled=false",
                "--/plugins/carb.tasking.plugin/threadCount=8", "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8",
                "--/persistent/physics/numThreads=4", "--/validate/p2p/enabled=false", "--/validate/iommu/enabled=false"]})
        simulate(app, args, cfg, status)
        status.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
        code = 0
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        write_json(args.output / "status.json", status)
        if app:
            app.app.post_quit(code)
            app.close()


if __name__ == "__main__":
    main()
