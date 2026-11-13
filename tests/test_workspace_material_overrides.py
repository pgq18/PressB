"""Appearance calibration must leave contacts and geometry unchanged."""
import math

import numpy as np
import pytest

pytest.importorskip("pxr.Usd")
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

from pressb.robots.piper.wrist_camera import apply_black_materials
from pressb.tasks.tabletop_cleanup.assets import MATERIALS, create_bin, create_object, material_settings


def _physical_snapshot(stage):
    """Ignore shaders, while comparing every authored geometry/physics value."""
    return {
        str(prim.GetPath()): {
            str(attr.GetName()): str(attr.Get())
            for attr in prim.GetAttributes()
            if attr.HasAuthoredValueOpinion() and str(attr.GetName()) != "primvars:displayColor"
        }
        for prim in stage.Traverse()
        if "/Looks" not in str(prim.GetPath())
    }


def _robot():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Piper")
    link = UsdGeom.Xform.Define(stage, "/World/Piper/link1")
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    UsdPhysics.MassAPI.Apply(link.GetPrim()).CreateMassAttr(1.2)
    UsdGeom.Xform.Define(stage, "/World/Piper/link1/visuals")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Piper/link1/visuals/mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    collision = UsdGeom.Cube.Define(stage, "/World/Piper/link1/collisions/cube")
    collision.CreateSizeAttr(.1)
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    return stage


def test_robot_override_changes_both_shaders_and_display_color_without_physics_changes():
    original, modified = _robot(), _robot()
    apply_black_materials(original)
    change = dict(color=[.01, .02, .03], roughness=.65, metallic=.1, specular=.12)
    result = apply_black_materials(modified, material_overrides=change)
    assert _physical_snapshot(original) == _physical_snapshot(modified)
    assert result["color_linear_rgb"] == change["color"]
    mdl = UsdShade.Shader(modified.GetPrimAtPath("/World/Looks/PiperGraphiteBlack/Mdl"))
    preview = UsdShade.Shader(modified.GetPrimAtPath("/World/Looks/PiperGraphiteBlack/Preview"))
    np.testing.assert_allclose(mdl.GetInput("diffuse_color_constant").Get(), change["color"])
    assert mdl.GetInput("reflection_roughness_constant").Get() == pytest.approx(.65)
    assert mdl.GetInput("metallic_constant").Get() == pytest.approx(.1)
    assert mdl.GetInput("specular_level").Get() == pytest.approx(.12)
    np.testing.assert_allclose(preview.GetInput("diffuseColor").Get(), change["color"])
    ior = preview.GetInput("ior").Get()
    assert ((ior - 1) / (ior + 1)) ** 2 == pytest.approx(.08 * .12)
    display = modified.GetPrimAtPath("/World/Piper/link1/visuals/mesh").GetAttribute("primvars:displayColor")
    np.testing.assert_allclose(display.Get()[0], change["color"])


def test_robot_default_and_empty_override_preserve_original_finish():
    original, empty = _robot(), _robot()
    apply_black_materials(original)
    apply_black_materials(empty, material_overrides={})
    assert original.GetRootLayer().ExportToString() == empty.GetRootLayer().ExportToString()
    mdl = UsdShade.Shader(original.GetPrimAtPath("/World/Looks/PiperGraphiteBlack/Mdl"))
    np.testing.assert_allclose(mdl.GetInput("diffuse_color_constant").Get(), [.018, .022, .028])
    assert mdl.GetInput("metallic_constant").Get() == pytest.approx(.22)
    assert mdl.GetInput("reflection_roughness_constant").Get() == pytest.approx(.37)
    assert not mdl.GetInput("specular_level")


@pytest.mark.parametrize("kind", ["gamepad", "speaker", "lidar", "bin"])
def test_cleanup_materials_change_only_appearance_not_contact_or_geometry(kind):
    baseline, changed, empty = (Usd.Stage.CreateInMemory() for _ in range(3))
    change = {
        name: dict(color=[.013, .015, .02], roughness=.63, metallic=.02, specular=.15)
        for name in MATERIALS
    }
    build = create_bin if kind == "bin" else lambda stage, path, **kw: create_object(stage, path, kind, **kw)
    baseline_meta = build(baseline, "/World/Prop")
    modified_meta = build(changed, "/World/Prop", material_overrides=change)
    build(empty, "/World/Prop", material_overrides={})
    assert baseline_meta == modified_meta
    assert _physical_snapshot(baseline) == _physical_snapshot(changed)
    assert baseline.GetRootLayer().ExportToString() == empty.GetRootLayer().ExportToString()
    for prim in changed.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        contact = UsdPhysics.MaterialAPI(prim)
        assert contact.GetStaticFrictionAttr().Get() == pytest.approx(.8)
        assert contact.GetDynamicFrictionAttr().Get() == pytest.approx(.65)
        assert contact.GetRestitutionAttr().Get() == 0
        mdl = UsdShade.Shader(changed.GetPrimAtPath(str(prim.GetPath()) + "/Mdl"))
        assert mdl.GetInput("reflection_roughness_constant").Get() == pytest.approx(.63)
        assert mdl.GetInput("specular_level").Get() == pytest.approx(.15)


@pytest.mark.parametrize("change", [
    {"color": [1, 2, 0]}, {"roughness": -1}, {"metallic": math.nan},
    {"specular": 2}, {"static_friction": .1}, {"mass_kg": .01}, [],
])
def test_visual_overrides_reject_invalid_and_physical_properties(change):
    with pytest.raises(ValueError):
        apply_black_materials(_robot(), material_overrides=change)
    with pytest.raises(ValueError):
        material_settings({"plastic": change})


@pytest.mark.parametrize("change", [{"plastci": {}}, [], "black"])
def test_cleanup_rejects_unknown_palette_names(change):
    with pytest.raises(ValueError, match="known material"):
        material_settings(change)


def test_cleanup_scene_forwards_palette_to_bin_and_dynamic_props():
    from pressb.tasks.tabletop_cleanup.configuration import load_config
    from pressb.tasks.tabletop_cleanup.scene import build_scene
    cfg = load_config({"scene_overrides": {"cleanup_materials": {
        "bin": {"roughness": .91}, "cover": {"color": [.02, .02, .02]},
    }}})
    stage = Usd.Stage.CreateInMemory()
    scene = build_scene(stage, cfg["scene"])
    root = scene["root_path"]
    shader = UsdShade.Shader(stage.GetPrimAtPath(root + "/Bin/Looks/bin/Mdl"))
    assert shader.GetInput("reflection_roughness_constant").Get() == pytest.approx(.91)
    for kind in ("gamepad", "speaker"):
        shader = UsdShade.Shader(stage.GetPrimAtPath(root + f"/Objects/{kind}/Looks/cover/Mdl"))
        np.testing.assert_allclose(shader.GetInput("diffuse_color_constant").Get(), [.02] * 3)
