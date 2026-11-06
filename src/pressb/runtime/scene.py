"""Shared USD material and geometry helpers for task scene builders.

Import after SimulationApp starts, or from an isolated USD Python interpreter.
These helpers author geometry and contact properties; they never control a task.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

try:
    from pxr import PhysxSchema
except ImportError:  # Plain USD inspection does not require Isaac Sim.
    PhysxSchema = None


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _vector(value, length, name):
    values = [float(v) for v in value]
    if len(values) != length or not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return values


def _unit(value, name):
    value = float(value)
    if not math.isfinite(value) or not 0. <= value <= 1.:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _color(value, name):
    return [_unit(v, name) for v in _vector(value, 3, name)]


def _material_settings(defaults, overrides, name):
    """Resolve numeric appearance only; contact parameters are never accepted."""
    if not isinstance(overrides, Mapping):
        raise ValueError(f"{name} must be a mapping of numeric material properties")
    allowed = {"color", "roughness", "metallic", "specular"}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"Unsupported {name} properties: {sorted(unknown)}; supported: {sorted(allowed)}")
    settings = {**defaults, **overrides}
    settings["color"] = _color(settings["color"], f"{name}.color (linear RGB)")
    settings["metallic"] = _unit(settings["metallic"], f"{name}.metallic")
    settings["roughness"] = _unit(settings["roughness"], f"{name}.roughness")
    if "specular" in overrides:
        settings["specular"] = _unit(overrides["specular"], f"{name}.specular")
        settings["specular_authored"] = True
    else:
        settings.setdefault("specular", .5)
        settings.setdefault("specular_authored", False)
    return settings


def _material(stage, path, color, *, metallic=0.0, roughness=0.6,
              static_friction=None, dynamic_friction=None, specular=None):
    """MDL for RTX plus PreviewSurface for ordinary USD viewers."""
    material = UsdShade.Material.Define(stage, path)
    mdl = UsdShade.Shader.Define(stage, path + "/Mdl")
    mdl.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    mdl.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    mdl.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    mdl.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mdl.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(metallic))
    mdl.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(roughness))
    if specular is not None:
        mdl.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(float(specular))
    mdl.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(mdl.ConnectableAPI(), "out")
    preview = UsdShade.Shader.Define(stage, path + "/Preview")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    if specular is not None:
        # OmniPBR uses normal reflectivity 0.08 weighted by specular_level.
        # PreviewSurface's metallic workflow exposes IOR instead: match its
        # dielectric F0 while retaining metallic reflectance and diffuse color.
        sqrt_f0 = math.sqrt(.08 * float(specular))
        preview.CreateInput("ior", Sdf.ValueTypeNames.Float).Set((1 + sqrt_f0) / (1 - sqrt_f0))
    material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
    if static_friction is not None:
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(float(static_friction))
        physics.CreateDynamicFrictionAttr(float(dynamic_friction))
        physics.CreateRestitutionAttr(0.0)
    return material


def _bind(prim, material, collision=False):
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(material)
    if collision:
        binding.Bind(material, materialPurpose="physics")
        UsdPhysics.CollisionAPI.Apply(prim)
        if PhysxSchema is not None:
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            api.CreateContactOffsetAttr(0.0005)
            api.CreateRestOffsetAttr(0.0)


def _cube(stage, path, center, size, material, *, collision=False, yaw_deg=0.):
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1.)
    shape.AddTranslateOp().Set(Gf.Vec3d(*center))
    if yaw_deg:
        shape.AddRotateZOp().Set(float(yaw_deg))
    shape.AddScaleOp().Set(Gf.Vec3f(*size))
    _bind(shape.GetPrim(), material, collision)
    return shape


def _cylinder_between(stage, path, start, end, radius, material):
    start, end = Gf.Vec3d(*start), Gf.Vec3d(*end)
    delta = end - start
    length = delta.GetLength()
    if length <= 0:
        raise ValueError("Cylinder endpoints must differ")
    shape = UsdGeom.Cylinder.Define(stage, path)
    shape.CreateAxisAttr("Z")
    shape.CreateHeightAttr(length)
    shape.CreateRadiusAttr(float(radius))
    shape.AddTranslateOp().Set((start + end) / 2.)
    rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), delta.GetNormalized()).GetQuat()
    shape.AddOrientOp().Set(Gf.Quatf(rotation))
    _bind(shape.GetPrim(), material)
    return shape


