#!/usr/bin/env python3
"""Compare appearance variants at recorded poses; no hardware or physics replay.

The layout is reset and settled once. During capture only the robot's measured
joint pose and visual shader/light properties change. This is not a grasp test.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    from pressb.registry import resolve
    from pressb.runtime.recording import write
    from pressb.sensors.color_response import apply_color_response

    node = json.loads(args.node.read_text())
    poses = json.loads(args.poses.read_text())
    variants = json.loads(args.variants.read_text())
    config = deepcopy(node["backend"]["config"])
    config.update(num_envs=1, gpu=args.gpu, headless=True, horizon=32,
                  terminate_on_success=False, log_dir=str(args.output / "simulation"))
    config["initial_pose"] = {"mode": "scene"}
    args.output.mkdir(parents=True, exist_ok=False)
    module, factory = resolve("task", node["backend"]["plugin"]).split(":")
    backend = getattr(importlib.import_module(module), factory)(config)
    try:
        from pxr import Gf, Sdf, UsdShade
        backend.reset([0], 0, {})
        home = backend.task.config["scene"]["initial_state_joint_rad_m"]
        backend.step([dict(env_id=0, actions=[home] * 30)])
        engine = backend.engine
        if any("camera_color_response" in patch for patch in variants.values()) and any(
                not response.identity for response in engine.camera_color_responses):
            raise ValueError("Response sweeps require an identity-response source scene; otherwise use an empty production variant")
        stage = engine.stage
        robot = engine.robot
        original = {}
        for prim in stage.Traverse():
            for attr in prim.GetAttributes():
                if attr.GetName() in {
                    "inputs:diffuse_color_constant", "inputs:diffuseColor",
                    "inputs:metallic_constant", "inputs:metallic",
                    "inputs:reflection_roughness_constant", "inputs:roughness",
                    "inputs:specular_level", "inputs:ior",
                    "inputs:intensity", "inputs:color", "inputs:width", "inputs:height",
                }:
                    original[str(attr.GetPath())] = attr.Get()
        report = dict(simulation_only=True, physics_replay=False, node=node,
                      resolved_scene=backend.task.config["scene"], poses=poses, variants=variants)
        frozen_time = engine.world.current_time
        cameras = {name: engine.relocate(0, path) for name, path in engine.template.camera_paths.items()}
        root = engine.relocate(0, backend.task.scene["root_path"])
        created = []
        for variant, patch in variants.items():
            for path, name in created:
                stage.GetPrimAtPath(path).RemoveProperty(name)
            created.clear()
            for path, value in original.items():
                stage.GetAttributeAtPath(path).Set(value)
            # All visual edits are restored from original before each variant.
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if not prim.IsA(UsdShade.Shader):
                    continue
                material = prim.GetParent().GetName()
                settings = patch.get("materials", {}).get(material, {}) if path.startswith(root + "/Looks/") else {}
                if material == "PiperGraphiteBlack":
                    settings = patch.get("robot_material", {})
                if "/Objects/" in path or "/Bin/" in path:
                    settings = patch.get("cleanup_materials", {}).get(material, {})
                for key, value in settings.items():
                    names = {"color": ("diffuse_color_constant", "diffuseColor"),
                             "roughness": ("reflection_roughness_constant", "roughness"),
                             "metallic": ("metallic_constant", "metallic"),
                             "specular": ("specular_level", "ior")}[key]
                    shader = UsdShade.Shader(prim)
                    for name in names:
                        prop = shader.GetInput(name)
                        if not prop and key == "specular" and ((prim.GetName() == "Mdl" and name == "specular_level") or (prim.GetName() == "Preview" and name == "ior")):
                            prop = shader.CreateInput(name, Sdf.ValueTypeNames.Float)
                            created.append((path, "inputs:" + name))
                        if prop:
                            actual = (1 + (.08 * value)**.5) / (1 - (.08 * value)**.5) if name == "ior" else value
                            prop.Set(Gf.Vec3f(*actual) if key == "color" else float(actual))
            lighting = patch.get("lighting", {})
            specs = [("Ambient", lighting.get("ambient", {}))]
            specs += [(s["name"], s) for s in lighting.get("area_lights", [])]
            for name, spec in specs:
                prim = stage.GetPrimAtPath(root + "/Lighting/" + name)
                for key in ("intensity", "color", "width", "height"):
                    if key in spec:
                        prim.GetAttribute("inputs:" + key).Set(Gf.Vec3f(*spec[key]) if key == "color" else float(spec[key]))
            for pose in poses:
                state = np.asarray(pose["observed"], dtype=float)
                joints = robot.commands.copy()
                joints[0, robot.dofs] = robot._full(state[None])[0]
                robot.arms.set_joint_positions(joints, robot.arm_rows)
                robot.arms.set_joint_velocities(np.zeros_like(joints), robot.arm_rows)
                robot.arms.set_joint_position_targets(joints, robot.arm_rows)
                engine.reset_render_history()
                images = None
                for _ in range(12):
                    images = engine._capture(subframes=8)
                for name, rgb in zip(engine.camera_names, images):
                    response = patch.get("camera_color_response", {}).get(name)
                    rgb = apply_color_response(rgb, response)
                    folder = args.output / variant
                    folder.mkdir(exist_ok=True)
                    Image.fromarray(rgb).save(folder / f"{pose['name']}_{name}.png")
                assert engine.world.current_time == frozen_time
                print(json.dumps(dict(variant=variant, pose=pose["name"], rendered=True)), flush=True)
        report.update(status="passed", physics_advanced_during_capture=False,
                      camera_paths=cameras, final_snapshot=backend.snapshot([0]))
        write(args.output / "report.json", report)
    except BaseException as error:
        write(args.output / "error.json", dict(error=f"{type(error).__name__}: {error}"))
        raise
    finally:
        backend.close()


if __name__ == "__main__":
    main()
