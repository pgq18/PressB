"""Configurable workspace twin built on the common laboratory workbench.

Object XY coordinates and static-prop XYZ centres are relative to the robot
base. The base may be raised by ``robot_base_height_above_table_m``; dynamic
object Z is always measured above the tabletop. Objects rest under gravity;
the scene does not attach, animate or teleport objects after construction.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping

from pxr import Gf, UsdPhysics

from pressb.runtime.scene import PhysxSchema, _color, _cube, _material, _positive, _unit, _vector
from pressb.runtime.workbench import build_workbench


def _specs(values, *, static):
    if not isinstance(values, list):
        raise ValueError("objects/static_props must be lists")
    result, names = [], set()
    allowed = {"name", "size", "color", "roughness", "metallic", "yaw_deg"}
    allowed |= {"center_base_m", "collision"} if static else {"center_xy", "mass", "clearance"}
    for value in values:
        if not isinstance(value, Mapping) or set(value) - allowed:
            raise ValueError(f"Unsupported playground geometry properties; expected {sorted(allowed)}")
        spec = dict(value)
        name = spec.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in names:
            raise ValueError("Object/prop names must be distinct USD identifiers")
        names.add(name)
        spec["size"] = [_positive(v, f"{name}.size") for v in _vector(spec.get("size", []), 3, f"{name}.size")]
        spec["color"] = _color(spec.get("color", [.5, .5, .5]), f"{name}.color")
        spec["roughness"] = _unit(spec.get("roughness", .6), f"{name}.roughness")
        spec["metallic"] = _unit(spec.get("metallic", 0.), f"{name}.metallic")
        spec["yaw_deg"] = float(spec.get("yaw_deg", 0.))
        if not math.isfinite(spec["yaw_deg"]):
            raise ValueError(f"{name}.yaw_deg must be finite")
        if static:
            spec["center_base_m"] = _vector(spec.get("center_base_m", []), 3, f"{name}.center_base_m")
            spec["collision"] = spec.get("collision", False)
            if type(spec["collision"]) is not bool:
                raise ValueError(f"{name}.collision must be boolean")
        else:
            spec["center_xy"] = _vector(spec.get("center_xy", []), 2, f"{name}.center_xy")
            spec["mass"] = _positive(spec.get("mass", .035), f"{name}.mass")
            spec["clearance"] = float(spec.get("clearance", .0005))
            if not math.isfinite(spec["clearance"]) or spec["clearance"] < 0:
                raise ValueError(f"{name}.clearance must be finite and nonnegative")
        result.append(spec)
    return result


def build_scene(stage, cfg=None):
    cfg = dict(cfg or {})
    cfg.setdefault("root_path", "/World/Playground")
    objects = _specs(cfg.get("objects", []), static=False)
    props = _specs(cfg.get("static_props", []), static=True)
    # Validate support before authoring the scene. Yaw affects the true footprint.
    table = cfg.get("table", {})
    xmin = float(cfg.get("table_x_min", table.get("x_min", -.12)))
    depth = float(cfg.get("table_depth", table.get("depth", 1.0)))
    width = float(cfg.get("table_width", table.get("width", 1.6)))
    for obj in objects:
        x, y = obj["center_xy"]
        a = math.radians(obj["yaw_deg"])
        sx, sy, _ = obj["size"]
        ex = (abs(math.cos(a)) * sx + abs(math.sin(a)) * sy) / 2
        ey = (abs(math.sin(a)) * sx + abs(math.cos(a)) * sy) / 2
        if not (xmin <= x-ex and x+ex <= xmin+depth and -width/2 <= y-ey and y+ey <= width/2):
            raise ValueError(f"Object {obj['name']} must be fully supported by the tabletop")
    scene = build_workbench(stage, cfg)
    root, base = scene["root_path"], scene["robot_base_position_world"]
    scene["objects"], scene["static_props"] = {}, {}
    for static, specs, collection in ((False, objects, "Objects"), (True, props, "StaticProps")):
        for spec in specs:
            name = spec["name"]
            material = _material(stage, f"{root}/Looks/{collection}_{name}", spec["color"],
                roughness=spec["roughness"], metallic=spec["metallic"],
                static_friction=.8, dynamic_friction=.65)
            local = (spec["center_base_m"] if static else
                     [*spec["center_xy"], spec["size"][2]/2 + spec["clearance"]])
            center = [float(b+p) for b, p in zip(base, local)]
            if not static:
                center[2] = float(scene["table_top_z"] + local[2])
            collision = spec["collision"] if static else True
            shape = _cube(stage, f"{root}/{collection}/{name}", center, spec["size"], material,
                          collision=collision, yaw_deg=spec["yaw_deg"])
            if collision:
                scene["collision_paths"].append(str(shape.GetPath()))
            if not static:
                body = UsdPhysics.RigidBodyAPI.Apply(shape.GetPrim())
                body.CreateRigidBodyEnabledAttr(True)
                body.CreateKinematicEnabledAttr(False)
                mass = UsdPhysics.MassAPI.Apply(shape.GetPrim())
                mass.CreateMassAttr(spec["mass"])
                x, y, z = spec["size"]
                m = spec["mass"] / 12
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(m*(y*y+z*z), m*(x*x+z*z), m*(x*x+y*y)))
                mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
                if PhysxSchema is not None:
                    api = PhysxSchema.PhysxRigidBodyAPI.Apply(shape.GetPrim())
                    api.CreateEnableCCDAttr(True)
                    api.CreateSolverPositionIterationCountAttr(16)
                    api.CreateSolverVelocityIterationCountAttr(4)
            key = "static_props" if static else "objects"
            scene[key][name] = {**spec, "path": str(shape.GetPath()), "center_world": center,
                                "collision": collision, "dynamic": not static}
    scene["task_id"] = "playground"
    scene["task_props_enabled"] = bool(objects)
    scene["outcome_semantics"] = "Workspace reconstruction only; no automatic task success or reward."
    return scene
