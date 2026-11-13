"""Actuator/reset contracts independent of an Isaac process or a task plugin."""
from collections import deque
import importlib.abc
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from pressb.control.pose_protocol import pose8_to_pose9
from pressb.robots.piper.parallel import PiperJointDriver, PiperPoseDriver


class Arms:
    names = ["joint8", "joint3", "aux", "joint1", "joint7", "joint6", "joint2", "joint4", "joint5"]

    def __init__(self):
        self.prim_paths = ["/World/envs/env_1/Piper/base_link", "/World/envs/env_0/Piper/base_link"]
        self.positions = np.zeros((2, 9))
        self.positions[:, 2] = [17, 19]
        self.velocities = np.zeros_like(self.positions)
        self.targets = self.positions.copy()
        self.stiffness = np.zeros_like(self.positions)
        self.damping = np.zeros_like(self.positions)
        self.view = self
        self.calls = []

    def get_dof_index(self, name):
        return self.names.index(name)

    def get_joint_positions(self):
        return self.positions.copy()

    def get_joint_velocities(self):
        return self.velocities.copy()

    def _write(self, key, value, indices):
        getattr(self, key)[indices] = value
        self.calls.append((key, np.asarray(value).copy(), np.asarray(indices).copy()))

    def set_joint_positions(self, value, indices):
        self._write("positions", value, indices)

    def set_joint_velocities(self, value, indices):
        self._write("velocities", value, indices)

    def set_joint_position_targets(self, value, indices):
        self._write("targets", value, indices)

    def get_dof_stiffnesses(self):
        return self.stiffness.copy()

    def get_dof_dampings(self):
        return self.damping.copy()

    def set_dof_stiffnesses(self, value, indices):
        self.stiffness[indices] = value[indices]

    def set_dof_dampings(self, value, indices):
        self.damping[indices] = value[indices]


@pytest.fixture
def setup(tmp_path):
    parts = ['<robot name="piper">']
    for i in range(1, 7):
        parent = 'base_link' if i == 1 else f'link{i-1}'
        axis = '0 0 1' if i in (1, 4, 6) else '0 1 0'
        parts.append(f'<joint name="joint{i}" type="revolute"><parent link="{parent}"/>'
                     f'<child link="link{i}"/><origin xyz="0.05 0 0.04"/>'
                     f'<axis xyz="{axis}"/><limit lower="-3.0" upper="3.0" velocity="2"/></joint>')
    parts.append('</robot>')
    urdf = tmp_path/'robot.urdf'; urdf.write_text(''.join(parts))
    scene = dict(robot_urdf=str(urdf), gripper_max_width_m=.08, arm_stiffness=12000.,
                 arm_damping=600., gripper_stiffness=1000., gripper_damping=80.)
    engine = SimpleNamespace(num_envs=2, robot_paths=[f'/World/envs/env_{i}/Piper' for i in range(2)],
                             config=dict(control_hz=30, physics_dt=1/120, max_episode_seconds=20),
                             backend=SimpleNamespace(ticks=np.array([12, 20])))
    initial = np.array([.1, .2, -.3, .4, -.5, .6, .04])
    return engine, scene, initial, Arms()


def test_lazy_imports_do_not_open_isaac_or_task_modules():
    source = str(Path(__file__).resolve().parents[1]/'src')
    code = '''
import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('isaacsim', 'pxr', 'omni', 'carb') or fullname.startswith('pressb.tasks.'):
            raise RuntimeError(fullname)
sys.meta_path.insert(0, Block())
from pressb.robots.piper.parallel import PiperJointDriver, PiperPoseDriver
from pressb.robots.piper.isaac import build_piper_runtime
'''
    result = subprocess.run([sys.executable, '-c', code, source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_joint_mapping_uses_real_tensor_rows_and_dof_names(setup):
    engine, scene, initial, arms = setup
    driver = PiperJointDriver(engine, scene, initial, arms=arms)
    np.testing.assert_array_equal(driver.arm_rows, [1, 0])
    np.testing.assert_allclose(driver.states(), np.repeat(initial[None], 2, axis=0))
    np.testing.assert_array_equal(arms.positions[:, 2], [17, 19])
    assert np.all(arms.stiffness[:, driver.arm_dofs] == 12000)
    assert np.all(arms.stiffness[:, driver.finger_dofs] == 1000)
    assert not arms.stiffness[:, 2].any()


def test_joint6_ema_is_per_slot_and_once_per_control_interval(setup):
    engine, scene, initial, arms = setup
    driver = PiperJointDriver(engine, scene, initial, action_dim=6,
                             fixed_gripper_width_m=.04, smoothing={'mode':'joint_ema', 'alpha':.5}, arms=arms)
    driver.cache_initial_state(); driver.reset([0, 1], {0:{}, 1:{}})
    requested = np.stack([initial[:6]+.2, initial[:6]-.2])
    applied = driver.begin_action([1, 0], requested)
    np.testing.assert_allclose(applied[1], np.r_[initial[:6]+.1, .04])
    np.testing.assert_allclose(applied[0], np.r_[initial[:6]-.1, .04])
    for tick in range(4):
        driver.before_tick([1, 0], tick, 4)
    assert [p.summary()['applied_actions'] for p in driver.postprocessors] == [1, 1]
    assert np.all(arms.targets[:, driver.finger_dofs] == [.02, -.02])
    driver.begin_action([0], np.array([initial[:6]+.1]))
    np.testing.assert_allclose(driver.last_sent[0, :6], initial[:6])
    np.testing.assert_allclose(driver.last_sent[1, :6], initial[:6]+.1)


def test_selected_reset_restores_settled_state_without_touching_peer(setup):
    engine, scene, initial, arms = setup
    driver = PiperJointDriver(engine, scene, initial, arms=arms)
    arms.positions[:, driver.arm_dofs] += .001
    driver.cache_initial_state()
    settled = driver.settled_joints.copy()
    driver.begin_action([0, 1], np.stack([initial+.01, initial+.02]))
    driver.before_tick([0, 1], 0, 4)
    arms.positions[:, driver.arm_dofs] += .1; arms.velocities[:] = .2
    peer = driver.diagnostic_state([1])
    driver.reset([0], {0:{}})
    np.testing.assert_allclose(arms.positions[driver.arm_rows[0]], settled[0])
    assert not arms.velocities[driver.arm_rows[0]].any()
    assert driver.diagnostic_state([1]) == peer
    assert engine.backend.ticks.tolist() == [12, 20]


def test_invalid_batch_does_not_partially_change_filters_or_targets(setup):
    engine, scene, initial, arms = setup
    driver = PiperJointDriver(engine, scene, initial, smoothing={'mode':'joint_ema'}, arms=arms)
    driver.cache_initial_state(); before = driver.diagnostic_state([0,1])
    invalid = np.stack([initial+.01, initial]); invalid[1, 2] = 10.
    with pytest.raises(ValueError, match='bounds'):
        driver.begin_action([0,1], invalid)
    assert before == driver.diagnostic_state([0,1])
    assert [p.summary()['applied_actions'] for p in driver.postprocessors] == [0,0]
    with pytest.raises(ValueError, match='bounds'):
        driver.reset([0,1], {0:{'initial_state_joint_rad_m':initial.tolist()},
                            1:{'initial_state_joint_rad_m':invalid[1].tolist()}})
    assert before == driver.diagnostic_state([0,1])


def pose_driver(setup):
    engine, scene, initial, arms = setup
    initial = initial.copy(); initial[6] = .008
    driver = PiperPoseDriver(engine, scene, initial, arms=arms, smoothing_window=3)
    return engine, scene, initial, arms, driver


def test_pose_reset_preserves_measured_cache_but_resets_solver_and_commands_home(setup):
    engine, scene, initial, arms, driver = pose_driver(setup)
    arms.positions[:, driver.arm_dofs] += .001
    arms.velocities[:, driver.arm_dofs] = .0002
    driver.cache_initial_state()
    cached = driver.settled_joints.copy(); cached_v = driver.settled_velocities.copy()
    driver.batched_controller.q[:] += .1
    arms.positions[:, driver.arm_dofs] += .2; arms.velocities[:] = .2
    peer = driver.diagnostic_state([1]); driver.reset([0], {0:{}})
    np.testing.assert_array_equal(arms.positions[driver.arm_rows[0]], cached[0])
    np.testing.assert_array_equal(arms.velocities[driver.arm_rows[0]], cached_v[0])
    np.testing.assert_allclose(driver.batched_controller.q[0], initial[:6])
    np.testing.assert_allclose(driver.commands[0,driver.arm_dofs], initial[:6])
    np.testing.assert_allclose(driver.history[0], np.repeat(initial[None,:6],3,axis=0))
    assert driver.diagnostic_state([1]) == peer


def test_pose_interpolation_matches_reference_four_tick_history(setup):
    engine, scene, initial, arms, driver = pose_driver(setup)
    driver.cache_initial_state(); driver.reset([0,1], {0:{},1:{}})
    observations = driver.observations([0,1])
    poses = pose8_to_pose9(np.array([observations[i]['state'] for i in [0,1]]))
    poses[:, 0] += .004
    previous = driver.batched_controller.q.copy()
    driver.validate_actions(poses)
    driver.begin_action([0,1], poses)
    assert not np.array_equal(driver.targets, previous)
    histories = [deque([initial[:6].copy() for _ in range(3)], maxlen=3) for _ in range(2)]
    for substep in range(4):
        for i in range(2):
            histories[i].append(previous[i] + (driver.targets[i]-previous[i])*(substep+1)/4)
        driver.before_tick([0,1], substep, 4)
        for i in range(2):
            np.testing.assert_allclose(arms.targets[driver.arm_rows[i],driver.arm_dofs], np.mean(histories[i],axis=0))
    measured = driver.observations([0,1])
    assert len(measured[0]['state']) == 8
    assert len(measured[0]['control_state']) == 19 + 6*3
    assert measured[0]['control_state'][-1] == pytest.approx((12/120)/20)
    assert measured[1]['control_state'][-1] == pytest.approx((20/120)/20)


def test_pose_sequence_is_prevalidated_before_any_ik_mutation(setup):
    _, _, _, _, driver = pose_driver(setup)
    driver.cache_initial_state(); driver.reset([0,1], {0:{},1:{}})
    valid = pose8_to_pose9(np.array([driver.observations([0])[0]['state']]))
    invalid = np.repeat(valid, 7, axis=0); invalid[-1,3:] = 0
    before = driver.diagnostic_state([0,1])
    with pytest.raises(ValueError, match='Degenerate'):
        driver.validate_actions(invalid)
    assert before == driver.diagnostic_state([0,1])


def test_authoring_mode_creates_no_camera_product_or_physics_reset(monkeypatch):
    from pressb.robots.piper import isaac
    class Node:
        def __getattr__(self, name):
            return lambda *a, **k: Node()
    stage = SimpleNamespace(GetPrimAtPath=lambda path: Node())
    world = SimpleNamespace(stage=stage, get_physics_context=lambda:Node())
    def module(name, **values):
        result = ModuleType(name); result.__dict__.update(values); monkeypatch.setitem(sys.modules,name,result)
    module('pxr', Gf=SimpleNamespace(Vec3d=lambda *x:x),
           Usd=SimpleNamespace(PrimRange=lambda prim:[]), UsdGeom=SimpleNamespace(Xformable=lambda prim:Node()),
           UsdPhysics=SimpleNamespace(MaterialAPI=SimpleNamespace(Apply=lambda prim:Node())),
           UsdShade=SimpleNamespace(Material=SimpleNamespace(Define=lambda *a:Node())), PhysxSchema=Node())
    module('isaacsim.core.api', World=lambda **k:world)
    module('isaacsim.core.utils.stage', add_reference_to_stage=lambda *a:None)
    module('carb', settings=SimpleNamespace(get_settings=lambda:Node()))
    module('pressb.robots.piper.wrist_camera', apply_black_materials=lambda *a, **k:None,
           attach_wrist_camera=lambda *a:{'camera_path':'/World/Piper/link6/WristCamera'})
    # Camera and articulation modules must never be imported in authoring mode.
    monkeypatch.setitem(sys.modules,'isaacsim.sensors.camera',None)
    monkeypatch.setitem(sys.modules,'isaacsim.core.prims',None)
    monkeypatch.setattr(isaac,'make_camera',lambda *a:'/World/Fixture/UpCamera')
    cfg=dict(initial_joint_positions_rad=[0]*6,initial_gripper_width_m=.04, physics_dt=1/120,fps=30,
             robot_usd='robot.usd',gripper_max_width_m=.08,finger_static_friction=.8,finger_dynamic_friction=.6,
             wrist_camera_resolution=[640,480],up_camera={'resolution':[640,480]})
    result=isaac.build_piper_runtime(cfg,lambda *a:{'root_path':'/World/Fixture',
        'robot_base_position_world':[0,0,.7]},asset_root='.',initialize=False)
    assert result.world is world and result.robot is None and result.cameras == {}
    assert result.robot_path == '/World/Piper'
    assert result.camera_paths == {'up':'/World/Fixture/UpCamera','wrist':'/World/Piper/link6/WristCamera'}
    np.testing.assert_allclose(result.initial_state,[0,0,0,0,0,0,.04])
