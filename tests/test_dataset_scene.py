"""CPU USD regression checks for cloned scene lighting and world joint anchors.

Run without starting Kit or allocating a GPU:
PYTHONPATH=.cache/usd-inspect:src .conda/envs/pressb/bin/python -m pytest tests/test_dataset_scene.py

Generate the default snapshot with ``bash scripts/run.sh --headless --gpu 0``
first, or set PRESSB_TEST_SNAPSHOT to an existing scene.usda. A fresh checkout
without a generated snapshot skips these integration tests; an explicitly
selected missing snapshot is an error.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if "PRESSB_TEST_SNAPSHOT" in os.environ:
    snapshot_setting = os.environ["PRESSB_TEST_SNAPSHOT"]
    if not snapshot_setting.strip():
        pytest.fail("PRESSB_TEST_SNAPSHOT must name an existing scene.usda", pytrace=False)
    SNAPSHOT = Path(snapshot_setting).expanduser().resolve()
    if not SNAPSHOT.is_file():
        pytest.fail(f"Explicit PRESSB_TEST_SNAPSHOT does not exist: {SNAPSHOT}", pytrace=False)
else:
    SNAPSHOT = ROOT / "outputs/edge_feedback/scene.usda"
    if not SNAPSHOT.is_file():
        pytest.skip(
            "Generated scene snapshot is absent; run bash scripts/run.sh --headless --gpu 0 "
            "or set PRESSB_TEST_SNAPSHOT to an existing scene.usda",
            allow_module_level=True,
        )

pytest.importorskip("pxr.Usd", reason="CPU USD package is required for scene composition checks")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

from pressb.dataset_scene import create_envs, set_panel_offset, validate_panel_layout


CONFIG = json.loads((ROOT / "configs/scene.json").read_text())


@pytest.fixture
def usd_world(monkeypatch):
    """Replace only Isaac runtime handles; compose the real snapshot in USD."""
    collision_calls = []

    class PrimHandle:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def get_world_pose(self):
            matrix = UsdGeom.Xformable(stage.GetPrimAtPath(self.prim_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            quaternion = matrix.ExtractRotationQuat()
            return np.asarray(matrix.ExtractTranslation()), np.r_[quaternion.GetReal(), quaternion.GetImaginary()]

        def set_world_pose(self, position, orientation):
            prim = stage.GetPrimAtPath(self.prim_path)
            inverse = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
            prim.GetAttribute("xformOp:translate").Set(inverse.Transform(Gf.Vec3d(*position)))
            self.orientation = np.array(orientation)

        def set_linear_velocity(self, velocity):
            self.linear_velocity = np.array(velocity)

        def set_angular_velocity(self, velocity):
            self.angular_velocity = np.array(velocity)

        def set_default_state(self, **kwargs):
            self.default_state = kwargs

    class Cloner:
        def __init__(self, stage):
            self.stage = stage

        def filter_collisions(self, *args):
            collision_calls.append(args)

    for module_name in ("isaacsim", "isaacsim.core", "isaacsim.core.cloner", "isaacsim.core.prims"):
        monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))
    sys.modules["isaacsim.core.cloner"].Cloner = Cloner
    sys.modules["isaacsim.core.prims"].SingleArticulation = PrimHandle
    sys.modules["isaacsim.core.prims"].SingleRigidPrim = PrimHandle

    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    return SimpleNamespace(
        stage=stage,
        scene=SimpleNamespace(add=lambda item: item),
        get_physics_context=lambda: SimpleNamespace(prim_path="/physicsScene"),
        collision_calls=collision_calls,
    )


@pytest.mark.parametrize("num_envs", [1, 2, 3])
def test_cloned_domes_are_inactive_without_changing_local_lights_or_anchors(usd_world, num_envs):
    source_hash = hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
    source = Usd.Stage.Open(str(SNAPSHOT))
    source_dome = UsdLux.DomeLight.Get(source, "/World/Lights/Ambient")
    assert source_dome.GetIntensityAttr().Get() == 845
    local_light_intensities = {
        str(prim.GetPath()).removeprefix("/World"): UsdLux.RectLight(prim).GetIntensityAttr().Get()
        for prim in source.Traverse() if prim.IsA(UsdLux.RectLight)
    }
    assert len(local_light_intensities) == 2

    envs = create_envs(usd_world, SNAPSHOT, num_envs, spacing=6., cfg=CONFIG)
    active_domes = [prim for prim in usd_world.stage.Traverse() if prim.IsA(UsdLux.DomeLight)]
    assert len(active_domes) == 1
    assert str(active_domes[0].GetPath()) == "/World/envs/env_0/Lights/Ambient"
    assert UsdLux.DomeLight(active_domes[0]).GetIntensityAttr().Get() == 845
    for env in envs:
        dome = usd_world.stage.GetPrimAtPath(env.prefix + "/Lights/Ambient")
        assert dome.IsValid()
        assert dome.IsActive() == (env.env_id == 0)
        # This must be a composition opinion on the clone, never an authored
        # intensity override or an edit to the referenced scene.
        dome_spec = usd_world.stage.GetRootLayer().GetPrimAtPath(dome.GetPath())
        if env.env_id:
            assert dome_spec.active is False
            assert "inputs:intensity" not in dome_spec.attributes
        for relative_path, intensity in local_light_intensities.items():
            local_light = UsdLux.RectLight.Get(usd_world.stage, env.prefix + relative_path)
            assert local_light.GetPrim().IsActive()
            assert local_light.GetIntensityAttr().Get() == intensity
        assert len(env.world_joint_anchors) == 13
        for anchor in env.world_joint_anchors:
            source_joint_path = "/World" + anchor["joint_path"].removeprefix(env.prefix)
            source_joint = UsdPhysics.Joint.Get(source, source_joint_path)
            side = anchor["world_side"]
            source_attr = source_joint.GetLocalPos0Attr() if side == 0 else source_joint.GetLocalPos1Attr()
            np.testing.assert_allclose(anchor["source_position"], source_attr.Get(), atol=1e-6)
            np.testing.assert_allclose(anchor["world_position"], np.asarray(source_attr.Get()) + env.offset,
                                       atol=1e-6)
        global_camera = UsdGeom.Xformable(usd_world.stage.GetPrimAtPath(env.prefix + "/GlobalCamera"))
        assert not global_camera.GetResetXformStack()
    assert len(usd_world.collision_calls) == 1
    assert usd_world.collision_calls[0][2] == [env.prefix for env in envs]
    assert hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() == source_hash
    assert source_dome.GetPrim().IsActive()
    assert source_dome.GetIntensityAttr().Get() == 845


def test_rejects_an_extra_active_global_dome(usd_world):
    UsdLux.DomeLight.Define(usd_world.stage, "/UnexpectedGlobalDome")
    with pytest.raises(ValueError, match="Expected only the shared DomeLight"):
        create_envs(usd_world, SNAPSHOT, 1, cfg=CONFIG)


def _matrix(stage, path):
    return np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))


@pytest.mark.parametrize("offset", [-.03, .021, .03])
def test_initial_panel_offset_moves_complete_geometry_and_only_panel_anchors(usd_world, offset):
    source = Usd.Stage.Open(str(SNAPSHOT))
    envs = create_envs(usd_world, SNAPSHOT, 2, cfg=dict(CONFIG, panel_offset_y_m=offset))
    for env in envs:
        for suffix in ("Frame", "Faceplate", "Header", "Footer", "Screw0", "Floor24/Bezel", "Floor35/Cap"):
            source_matrix = _matrix(source, "/World/Panel/" + suffix)
            expected = source_matrix.copy()
            expected[3, :3] += env.offset + [0., offset, 0.]
            np.testing.assert_allclose(_matrix(usd_world.stage, env.prefix + "/Panel/" + suffix), expected, atol=1e-6)
        for anchor in env.world_joint_anchors:
            expected = np.asarray(anchor["source_position"]) + env.offset
            if anchor["is_panel_anchor"]:
                expected[1] += offset
            np.testing.assert_allclose(anchor["world_position"], expected, atol=1e-7)
        evidence = validate_panel_layout(usd_world, env)
        assert evidence["panel_offset_y_m"] == offset
        assert evidence["global_camera"]["all_inside"]


def test_repeated_live_moves_reset_caps_and_keep_other_environment_unchanged(usd_world):
    source_hash = hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
    env, neighbor = create_envs(usd_world, SNAPSHOT, 2, cfg=CONFIG)
    neighbor_before = {floor: body.get_world_pose()[0].copy() for floor, body in neighbor.buttons.items()}
    anchors_before = {a["joint_path"]: list(a["world_position"]) for a in neighbor.world_joint_anchors}
    for offset in (.03, -.03, .01, .01, 0.):
        cap = usd_world.stage.GetPrimAtPath(env.button_info[24]["body_path"])
        translation = cap.GetAttribute("xformOp:translate")
        translation.Set(translation.Get() + Gf.Vec3d(.003, .001, 0.))
        set_panel_offset(usd_world, env, offset)
        evidence = validate_panel_layout(usd_world, env)
        assert evidence["fixed_rig_unchanged"]
        assert evidence["panel_offset_y_m"] == offset
        assert evidence["max_button_rest_error_m"] < 1e-8
        assert evidence["max_spring_anchor_error_m"] < 1e-6
        assert np.mean(np.asarray(evidence["panel_world_bounds_m"]), axis=0)[1] == pytest.approx(offset)
        for floor, body in env.buttons.items():
            np.testing.assert_array_equal(body.linear_velocity, np.zeros(3))
            np.testing.assert_array_equal(body.angular_velocity, np.zeros(3))
            np.testing.assert_allclose(body.default_state["position"], env.button_info[floor]["body_center"])
            assert not env.light_inputs[floor]["illuminated"].Get()
        for floor, body in neighbor.buttons.items():
            np.testing.assert_array_equal(body.get_world_pose()[0], neighbor_before[floor])
        for anchor in neighbor.world_joint_anchors:
            assert anchor["world_position"] == anchors_before[anchor["joint_path"]]
    assert hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() == source_hash


def test_validation_rejects_displaced_cap_or_world_anchor(usd_world):
    env = create_envs(usd_world, SNAPSHOT, 1, cfg=CONFIG)[0]
    set_panel_offset(usd_world, env, .015)
    position, orientation = env.buttons[24].get_world_pose()
    env.buttons[24].set_world_pose(position + [.001, 0., 0.], orientation)
    with pytest.raises(RuntimeError, match="did not settle"):
        validate_panel_layout(usd_world, env)
    set_panel_offset(usd_world, env, .015)
    anchor = next(a for a in env.world_joint_anchors if a["is_panel_anchor"])
    UsdPhysics.Joint.Get(usd_world.stage, anchor["joint_path"]).GetLocalPos0Attr().Set(Gf.Vec3f(0., 0., 0.))
    with pytest.raises(RuntimeError, match="did not settle"):
        validate_panel_layout(usd_world, env)


def test_validation_rejects_cropped_full_panel_or_changed_camera_configuration(usd_world):
    env = create_envs(usd_world, SNAPSHOT, 1, cfg=CONFIG)[0]
    set_panel_offset(usd_world, env, .5)
    with pytest.raises(RuntimeError, match="does not fit fixed camera"):
        validate_panel_layout(usd_world, env)
    set_panel_offset(usd_world, env, 0.)
    env.config["global_camera_eye"] = [-.38, .24, 1.05]
    with pytest.raises(RuntimeError, match="optical pose disagrees"):
        validate_panel_layout(usd_world, env)


@pytest.mark.parametrize("offset", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_panel_move_is_rejected_before_scene_mutation(usd_world, offset):
    env = create_envs(usd_world, SNAPSHOT, 1, cfg=CONFIG)[0]
    before = usd_world.stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match="must be finite"):
        set_panel_offset(usd_world, env, offset)
    assert usd_world.stage.GetRootLayer().ExportToString() == before


def test_xy_moves_keep_back_wall_attached_without_moving_side_wall_or_table(usd_world):
    from pressb.scene import BACK_WALL_COMPONENTS
    env = create_envs(usd_world, SNAPSHOT, 1, cfg=CONFIG)[0]
    wall_original = {name: _matrix(usd_world.stage, env.prefix + "/Environment/" + name)
                     for name in BACK_WALL_COMPONENTS}
    fixed_paths = [env.prefix + "/Environment/" + name for name in ("SideWall", "TableCollision", "Floor")]
    fixed_original = {path: _matrix(usd_world.stage, path) for path in fixed_paths}
    for x, y in ((.01, .02), (-.01, -.02), (.006, -.014), (0., 0.)):
        set_panel_offset(usd_world, env, y, x)
        evidence = validate_panel_layout(usd_world, env)
        assert evidence["panel_offset_x_m"] == x
        assert evidence["back_wall_alignment_error_m"] < 1e-7
        for floor in range(24, 36):
            assert env.button_info[floor]["rest_x"] == pytest.approx(CONFIG["button_face_x"] + x + .003)
            assert evidence["button_rest_positions_world_m"][str(floor)][0] == pytest.approx(CONFIG["button_face_x"] + x + .003)
        for name, original in wall_original.items():
            expected = original.copy()
            expected[3, 0] += x
            np.testing.assert_allclose(_matrix(usd_world.stage, env.prefix + "/Environment/" + name), expected, atol=1e-8)
        for path, expected in fixed_original.items():
            np.testing.assert_array_equal(_matrix(usd_world.stage, path), expected)


def test_nonzero_static_snapshot_offsets_are_not_applied_twice(usd_world, tmp_path):
    from pressb.scene import set_back_wall_offset
    snapshot = tmp_path / "offset_source.usda"
    source = Usd.Stage.CreateNew(str(snapshot))
    UsdGeom.Xform.Define(source, "/World").GetPrim().GetReferences().AddReference(str(SNAPSHOT), "/World")
    x, y = -.006, .013
    panel = UsdGeom.Xformable(source.GetPrimAtPath("/World/Panel"))
    # Current scene snapshots may already contain the panel offset transform.
    offset_attr = panel.GetPrim().GetAttribute("xformOp:translate:panelOffset")
    if offset_attr:
        offset_attr.Set(Gf.Vec3d(x, y, 0.))
    else:
        panel.AddTranslateOp(opSuffix="panelOffset").Set(Gf.Vec3d(x, y, 0.))
    panel.GetPrim().CreateAttribute("pressb:panelOffsetXM", Sdf.ValueTypeNames.Double).Set(x)
    panel.GetPrim().CreateAttribute("pressb:panelOffsetYM", Sdf.ValueTypeNames.Double).Set(y)
    set_back_wall_offset(source, "/World/Environment", x)
    for floor in range(24, 36):
        joint = UsdPhysics.Joint.Get(source, f"/World/Panel/Floor{floor}/SpringJoint")
        attribute = joint.GetLocalPos0Attr()
        attribute.Set(attribute.Get() + Gf.Vec3f(x, y, 0.))
    source.GetRootLayer().Save()
    before = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    env = create_envs(usd_world, snapshot, 1, cfg=dict(CONFIG, panel_offset_x_m=.008, panel_offset_y_m=-.02))[0]
    assert validate_panel_layout(usd_world, env)["panel_offset_x_m"] == .008
    for target_x, target_y in ((-.01, -.01), (.01, .02), (0., 0.)):
        set_panel_offset(usd_world, env, target_y, target_x)
        evidence = validate_panel_layout(usd_world, env)
        assert evidence["panel_offset_x_m"] == target_x
        assert evidence["panel_offset_y_m"] == target_y
        assert evidence["max_spring_anchor_error_m"] < 1e-6
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == before


def test_scene_builder_static_xy_offset_matches_button_metadata_and_world_anchors(monkeypatch):
    from pressb import scene
    monkeypatch.setattr(scene, "_table", lambda *args: {})
    # Avoid touching the generated label files during CPU geometry tests.
    monkeypatch.setattr(scene, "_label", lambda *args: None)
    stage = Usd.Stage.CreateInMemory()
    x, y = -.008, .023
    result = scene.build_scene(stage, dict(CONFIG, panel_offset_x_m=x, panel_offset_y_m=y))
    for floor, info in result["buttons"].items():
        expected = np.array([CONFIG["button_face_x"] + x + .003,
                             (1 if floor < 30 else -1) * CONFIG["button_column_y"] + y,
                             CONFIG["button_bottom_z"] + (floor - 24) % 6 * CONFIG["button_pitch_z"]])
        np.testing.assert_allclose(info["body_center"], expected, atol=1e-8)
        np.testing.assert_allclose(_matrix(stage, info["body_path"])[3, :3], expected, atol=1e-8)
        np.testing.assert_allclose(UsdPhysics.Joint.Get(stage, info["joint_path"]).GetLocalPos0Attr().Get(), expected, atol=1e-7)
        assert info["rest_x"] == pytest.approx(expected[0])
    wall_center = _matrix(stage, "/World/Environment/Wall")[3, :3]
    assert wall_center[0] == pytest.approx(CONFIG["button_face_x"] + .06 + .025 + x)
