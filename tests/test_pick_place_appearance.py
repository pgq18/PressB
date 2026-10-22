"""Appearance controls must preserve the reconstructed task's contact physics.

CPU-only USD checks; no SimulationApp or GPU is needed. Run with
``PYTHONPATH=.cache/usd-inspect:src .conda/envs/pressb/bin/python -m pytest``.
"""

import copy

import numpy as np
import pytest

Usd = pytest.importorskip("pxr.Usd")
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, UsdShade

from pressb.pick_place.scene import build_scene


def scene(config=None):
    stage = Usd.Stage.CreateInMemory()
    result = build_scene(stage, config or {})
    return stage, result


def bound_material(stage, path, purpose=""):
    material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial(
        materialPurpose=purpose
    )
    assert material
    return material


def shader_value(stage, material, shader, name):
    return UsdShade.Shader(stage.GetPrimAtPath(str(material.GetPath()) + "/" + shader)).GetInput(name).Get()


def physical_state(stage, result):
    """Compare effective contact properties, even when visual bindings change."""
    state = {}
    for path in result["collision_paths"]:
        prim = stage.GetPrimAtPath(path)
        attributes = {
            attr.GetName(): str(attr.Get())
            for attr in prim.GetAttributes()
            if attr.GetName().startswith(("physics:", "physx", "xformOp:"))
            or attr.GetName() in {"size", "height", "radius", "axis", "xformOpOrder"}
        }
        material = bound_material(stage, path, "physics")
        contact = UsdPhysics.MaterialAPI(material.GetPrim())
        assert contact
        state[path] = (
            prim.GetTypeName(), tuple(prim.GetAppliedSchemas()), attributes,
            contact.GetStaticFrictionAttr().Get(), contact.GetDynamicFrictionAttr().Get(),
            contact.GetRestitutionAttr().Get(),
        )
    return state


def test_empty_appearance_configuration_preserves_legacy_usd():
    config = {"material_colors": {"Mat": [.02, .18, .09]},
              "ambient_intensity": 270., "key_intensity": 1300.}
    legacy, _ = scene(config)
    explicit, _ = scene({**config, "materials": {}, "lighting": {}})
    assert legacy.GetRootLayer().ExportToString() == explicit.GetRootLayer().ExportToString()


def test_material_overrides_change_rendering_without_changing_contact_physics():
    config = {"object": {"shape": "cube", "edge": .035, "mass": .027,
                         "center_xy": [.31, -.06], "yaw_deg": 19.}}
    baseline, original = scene(config)
    updated = copy.deepcopy(config)
    updated.update(material_colors={"Mat": [.01, .12, .08]},
                   materials={"Mat": {"color": [.04, .21, .13], "roughness": .34,
                                      "metallic": .1, "specular": .3},
                              "Cardboard": {"roughness": .89}})
    updated["object"].update(material={"color": [.08, .12, .21], "roughness": .27,
                                       "metallic": .2, "specular": .4})
    changed, result = scene(updated)
    assert physical_state(baseline, original) == physical_state(changed, result)
    assert original["object_geometry"] == result["object_geometry"]
    table = bound_material(changed, result["root_path"] + "/Table/Top")
    np.testing.assert_allclose(shader_value(changed, table, "Preview", "diffuseColor"), [.04, .21, .13])
    assert shader_value(changed, table, "Preview", "roughness") == pytest.approx(.34)
    assert shader_value(changed, table, "Mdl", "reflection_roughness_constant") == pytest.approx(.34)
    assert shader_value(changed, table, "Preview", "metallic") == pytest.approx(.1)
    assert shader_value(changed, table, "Mdl", "specular_level") == pytest.approx(.3)
    ior = shader_value(changed, table, "Preview", "ior")
    assert ((ior - 1) / (ior + 1)) ** 2 == pytest.approx(.08 * .3)
    obj = bound_material(changed, result["object_path"])
    np.testing.assert_allclose(shader_value(changed, obj, "Preview", "diffuseColor"), [.08, .12, .21])
    assert shader_value(changed, obj, "Preview", "roughness") == pytest.approx(.27)
    assert result["appearance"]["materials"]["Mat"]["specular"] == pytest.approx(.3)


def test_legacy_object_color_takes_precedence_without_discarding_surface_properties():
    config = {"object": {"shape": "cube", "edge": .035, "color": [.55, .03, .02],
                         "material": {"color": [.01, .1, .4], "roughness": .31,
                                      "metallic": .2, "specular": .45}}}
    stage, result = scene(config)
    material = bound_material(stage, result["object_path"])
    np.testing.assert_allclose(shader_value(stage, material, "Preview", "diffuseColor"), [.55, .03, .02])
    assert shader_value(stage, material, "Preview", "roughness") == pytest.approx(.31)
    assert shader_value(stage, material, "Preview", "metallic") == pytest.approx(.2)


def test_area_lights_point_at_world_targets_with_nonzero_robot_base_offset():
    base = np.array([.23, -.41, .84])
    lamps = [
        {"name": "Side", "position_base_m": [.7, -.5, .8],
         "target_base_m": [.3, .1, .04], "width": .6, "height": .4,
         "intensity": 820., "color": [.9, .95, 1.]},
        {"name": "Top", "position_base_m": [.25, .1, .9],
         "target_base_m": [.25, .1, .02], "width": .8, "height": .7,
         "intensity": 1100., "color": [1., .95, .9]},
    ]
    config = {"robot_base_xy": base[:2].tolist(), "table_height": base[2],
              "lighting": {"ambient": {"intensity": 120., "color": [.8, .9, 1.]},
                           "area_lights": lamps}}
    stage, result = scene(config)
    assert result["appearance"]["lighting"]["enabled"]
    root = result["root_path"]
    for lamp in lamps:
        prim = stage.GetPrimAtPath(root + "/Lighting/" + lamp["name"])
        assert prim.IsA(UsdLux.RectLight)
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        position = np.array(transform.Transform(Gf.Vec3d(0, 0, 0)))
        ray = np.array(transform.TransformDir(Gf.Vec3d(0, 0, -1)))
        np.testing.assert_allclose(position, base + lamp["position_base_m"], atol=1e-7)
        direction = base + lamp["target_base_m"] - position
        np.testing.assert_allclose(ray / np.linalg.norm(ray), direction / np.linalg.norm(direction), atol=1e-6)
        light = UsdLux.RectLight(prim)
        assert light.GetWidthAttr().Get() == pytest.approx(lamp["width"])
        assert light.GetHeightAttr().Get() == pytest.approx(lamp["height"])
        assert light.GetIntensityAttr().Get() == pytest.approx(lamp["intensity"])
        np.testing.assert_allclose(light.GetColorAttr().Get(), lamp["color"])
    ambient = UsdLux.DomeLight(stage.GetPrimAtPath(root + "/Lighting/Ambient"))
    assert ambient.GetIntensityAttr().Get() == pytest.approx(120.)
    np.testing.assert_allclose(ambient.GetColorAttr().Get(), [.8, .9, 1.])
    assert len([prim for prim in stage.Traverse() if prim.IsA(UsdLux.RectLight)]) == 2

    dark, dark_result = scene({**config, "lights": False})
    assert not dark_result["appearance"]["lighting"]["enabled"]
    assert not any(prim.HasAPI(UsdLux.LightAPI) for prim in dark.Traverse())


@pytest.mark.parametrize("config", [
    {"materials": {"Mat": {"roughness": -0.1}}},
    {"materials": {"Cardboard": {"color": [1.1, .2, .3]}}},
    {"object": {"shape": "cube", "edge": .03, "material": {"roughness": 1.1}}},
    {"lighting": {"ambient": {"intensity": -1.}}},
    {"lighting": {"ambient": {"intensity": float("nan")}}},
    {"lighting": {"ambient": {"color": [.9, -.1, 1.]}}},
])
def test_invalid_appearance_values_are_rejected(config):
    with pytest.raises(ValueError):
        scene(config)


def test_optional_cardboard_outer_changes_only_four_outward_visual_surfaces():
    config = {"box": {"size": [.10, .15, .06], "yaw_deg": 17},
              "materials": {"Cardboard": {"color": [.48, .24, .10], "roughness": .86}}}
    original_stage, original = scene(config)
    assert not original_stage.GetPrimAtPath(original["root_path"] + "/Looks/CardboardOuter")
    assert not original_stage.GetPrimAtPath(original["box_path"] + "/OuterAppearance")
    changed_config = copy.deepcopy(config)
    changed_config["materials"]["CardboardOuter"] = {"color": [.20, .17, .14]}
    changed_stage, changed = scene(changed_config)
    assert physical_state(original_stage, original) == physical_state(changed_stage, changed)
    assert original["box_inside"] == changed["box_inside"]
    assert original["collision_paths"] == changed["collision_paths"]
    info = changed["appearance"]["box_outer_surfaces"]
    assert info["enabled"] and info["visual_only"]
    assert len(info["paths"]) == 4
    assert info["offset_m"] == pytest.approx(.00005)
    for path in info["paths"]:
        prim = changed_stage.GetPrimAtPath(path)
        assert not prim.HasAPI(UsdPhysics.CollisionAPI)
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        mesh = UsdGeom.Mesh(prim)
        assert list(mesh.GetFaceVertexCountsAttr().Get()) == [4]
        assert not mesh.GetDoubleSidedAttr().Get()
        points = np.asarray(mesh.GetPointsAttr().Get())
        normal = np.asarray(mesh.GetNormalsAttr().Get())[0]
        # Winding and normals point out of the cavity on all four sides.
        assert np.dot(np.cross(points[1]-points[0], points[2]-points[0]), normal) > 0
        assert np.dot(points.mean(axis=0)[:2], normal[:2]) > 0
        material = bound_material(changed_stage, path)
        assert str(material.GetPath()) == info["material_path"]
        np.testing.assert_allclose(shader_value(changed_stage, material, "Preview", "diffuseColor"), [.20, .17, .14])
        assert shader_value(changed_stage, material, "Preview", "roughness") == pytest.approx(.86)
        assert not material.GetPrim().HasAPI(UsdPhysics.MaterialAPI)
    # Delete only the optional visual skin and material: original USD remains
    # byte-identical, including its warm inner walls and all collision settings.
    changed_stage.RemovePrim(changed["box_path"] + "/OuterAppearance")
    changed_stage.RemovePrim(info["material_path"])
    assert original_stage.GetRootLayer().ExportToString() == changed_stage.GetRootLayer().ExportToString()
