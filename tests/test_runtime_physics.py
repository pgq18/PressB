"""CPU contracts for shared PhysX adapters and their legacy import aliases."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.runtime.physics import (
    PhysXErrorGuard,
    TensorArms,
    TensorRigidBodies,
    path_rows,
    view_env_order,
)


def test_physics_import_does_not_load_isaac_or_task_backends():
    source = str(Path(__file__).resolve().parents[1] / "src")
    script = """
import importlib.abc
import sys

sys.path.insert(0, sys.argv[1])
blocked = ("isaacsim", "omni", "pxr", "carb", "pressb.online_rl", "pressb.rl_nodes")

class RejectSimulationImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == root or fullname.startswith(root + ".") for root in blocked):
            raise AssertionError("Unexpected simulation import: " + fullname)

sys.meta_path.insert(0, RejectSimulationImports())
from pressb.runtime.physics import (
    PhysXErrorGuard, TensorArms, TensorRigidBodies, path_rows, view_env_order,
)
assert all(value is not None for value in (
    PhysXErrorGuard, TensorArms, TensorRigidBodies, path_rows, view_env_order,
))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, source],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_fast_simulation_names_are_shared_component_aliases():
    from pressb.online_rl import fast_simulation

    assert fast_simulation.PhysXErrorGuard is PhysXErrorGuard
    assert fast_simulation._TensorArms is TensorArms
    assert fast_simulation._TensorButtons is TensorRigidBodies
    assert fast_simulation.view_env_order is view_env_order


class FakeArticulationView:
    def __init__(self):
        self.shared_metatype = SimpleNamespace(dof_names=["joint1", "joint2", "finger"])
        self.positions = np.arange(12, dtype=np.float32).reshape(4, 3)
        self.velocities = self.positions.copy() + 20
        self.targets = self.positions.copy() + 40
        self.writes = []

    def get_dof_positions(self):
        return self.positions

    def get_dof_velocities(self):
        return self.velocities

    def get_dof_position_targets(self):
        return self.targets

    def _write(self, attribute, values, indices):
        original = getattr(self, attribute)
        assert not np.shares_memory(values, original)
        assert indices.dtype == np.uint32
        assert values.shape == original.shape
        self.writes.append((attribute, values.copy(), indices.copy(), original.copy()))
        original[indices] = values[indices]

    def set_dof_positions(self, values, indices):
        self._write("positions", values, indices)

    def set_dof_velocities(self, values, indices):
        self._write("velocities", values, indices)

    def set_dof_position_targets(self, values, indices):
        self._write("targets", values, indices)


@pytest.mark.parametrize("method,attribute", [
    ("set_joint_positions", "positions"),
    ("set_joint_velocities", "velocities"),
    ("set_joint_position_targets", "targets"),
])
def test_arm_updates_keep_requested_order_and_leave_other_slots_unchanged(method, attribute):
    arms = object.__new__(TensorArms)
    arms.view = FakeArticulationView()
    before = {name: getattr(arms.view, name).copy() for name in ("positions", "velocities", "targets")}
    indices = np.array([3, 1], dtype=np.int64)
    values = np.array([[101, 102, 103], [201, 202, 203]], dtype=np.float32)

    getattr(arms, method)(values, indices=indices)

    expected = before[attribute].copy()
    expected[indices] = values
    np.testing.assert_array_equal(getattr(arms.view, attribute), expected)
    written_attribute, written_values, written_indices, original_at_write = arms.view.writes[0]
    assert written_attribute == attribute
    np.testing.assert_array_equal(written_values, expected)
    np.testing.assert_array_equal(written_indices, indices)
    np.testing.assert_array_equal(original_at_write, before[attribute])
    for name in before.keys() - {attribute}:
        np.testing.assert_array_equal(getattr(arms.view, name), before[name])


def test_arm_readers_and_dof_lookup_use_view_order():
    arms = object.__new__(TensorArms)
    arms.view = FakeArticulationView()

    np.testing.assert_array_equal(arms.get_joint_positions(), arms.view.positions)
    np.testing.assert_array_equal(arms.get_joint_velocities(), arms.view.velocities)
    assert arms.get_dof_index("finger") == 2
    assert arms.get_dof_index("joint1") == 0


class FakeRigidBodyView:
    def __init__(self, paths):
        self.prim_paths = paths
        self.transforms = np.array([
            [1, 2, 3, .1, .2, .3, .9],
            [4, 5, 6, .4, .5, .6, .7],
            [7, 8, 9, .8, .7, .6, .5],
        ], dtype=np.float32)
        self.velocities = np.arange(18, dtype=np.float32).reshape(3, 6)
        self.writes = []

    def get_transforms(self):
        return self.transforms

    def get_velocities(self):
        return self.velocities

    def _write(self, attribute, values, indices):
        original = getattr(self, attribute)
        assert not np.shares_memory(values, original)
        assert indices.dtype == np.uint32
        assert values.shape == original.shape
        self.writes.append((attribute, values.copy(), indices.copy(), original.copy()))
        original[indices] = values[indices]

    def set_transforms(self, values, indices):
        self._write("transforms", values, indices)

    def set_velocities(self, values, indices):
        self._write("velocities", values, indices)


@pytest.fixture
def bodies():
    paths = [f"/World/env_{index}/Object" for index in (2, 0, 1)]
    view = FakeRigidBodyView(paths)
    calls = []

    def create_rigid_body_view(requested):
        calls.append(requested)
        return view

    adapter = TensorRigidBodies(SimpleNamespace(create_rigid_body_view=create_rigid_body_view), paths)
    assert calls == [paths]
    assert adapter.prim_paths == paths
    return adapter


def test_body_reads_convert_xyzw_to_wxyz_and_return_independent_pose_buffers(bodies):
    before = bodies.view.transforms.copy()
    positions, orientations = bodies.get_world_poses()

    np.testing.assert_array_equal(positions, before[:, :3])
    np.testing.assert_array_equal(orientations, before[:, [6, 3, 4, 5]])
    positions[:] = -100
    orientations[:] = -200
    np.testing.assert_array_equal(bodies.view.transforms, before)
    np.testing.assert_array_equal(bodies.get_velocities(), bodies.view.velocities)


def test_body_pose_updates_convert_wxyz_to_xyzw_for_unsorted_subset(bodies):
    before = bodies.view.transforms.copy()
    velocities = bodies.view.velocities.copy()
    indices = np.array([2, 0], dtype=np.int64)
    positions = np.array([[10, 11, 12], [13, 14, 15]], dtype=np.float32)
    orientations = np.array([[.5, .1, .2, .3], [.7, .4, .5, .6]], dtype=np.float32)

    bodies.set_world_poses(positions, orientations, indices=indices)

    expected = before.copy()
    expected[indices] = np.column_stack((positions, orientations[:, [1, 2, 3, 0]]))
    np.testing.assert_array_equal(bodies.view.transforms, expected)
    np.testing.assert_array_equal(bodies.view.velocities, velocities)
    attribute, written, written_indices, original_at_write = bodies.view.writes[0]
    assert attribute == "transforms"
    np.testing.assert_array_equal(written, expected)
    np.testing.assert_array_equal(written_indices, indices)
    np.testing.assert_array_equal(original_at_write, before)
    new_positions, new_orientations = bodies.get_world_poses()
    np.testing.assert_array_equal(new_positions[indices], positions)
    np.testing.assert_array_equal(new_orientations[indices], orientations)


def test_body_velocity_updates_preserve_unselected_slots_and_poses(bodies):
    before = bodies.view.velocities.copy()
    poses = bodies.view.transforms.copy()
    indices = np.array([2, 0], dtype=np.int64)
    values = np.arange(12, dtype=np.float32).reshape(2, 6) + 100

    bodies.set_velocities(values, indices=indices)

    expected = before.copy()
    expected[indices] = values
    np.testing.assert_array_equal(bodies.get_velocities(), expected)
    np.testing.assert_array_equal(bodies.view.transforms, poses)
    attribute, written, written_indices, original_at_write = bodies.view.writes[0]
    assert attribute == "velocities"
    np.testing.assert_array_equal(written, expected)
    np.testing.assert_array_equal(written_indices, indices)
    np.testing.assert_array_equal(original_at_write, before)


def test_view_env_order_accepts_path_objects_and_environment_roots():
    paths = [Path("/World/env_2"), Path("/World/env_0/Robot"), Path("/World/env_1")]

    np.testing.assert_array_equal(view_env_order(paths, 3), [1, 2, 0])
    assert view_env_order([], 0).size == 0


def test_path_rows_maps_view_paths_to_requested_order_as_int64():
    paths = [Path("/World/env_1/Object"), Path("/World/env_2/Object"), Path("/World/env_0/Object")]
    requested = ["/World/env_2/Object", "/World/env_0/Object", "/World/env_1/Object"]

    rows = path_rows(paths, requested)

    assert rows.dtype == np.int64
    np.testing.assert_array_equal(rows, [1, 2, 0])
    assert path_rows([], []).dtype == np.int64
    assert path_rows([], []).size == 0


@pytest.mark.parametrize("paths,requested", [
    (["a"], ["a", "b"]),
    (["a", "a"], ["a", "b"]),
    (["a", "c"], ["a", "b"]),
    (["a", "b", "c"], ["a", "b"]),
    (["a", "b"], ["a", "a"]),
])
def test_path_rows_keeps_runtime_error_for_incomplete_or_unexpected_mapping(paths, requested):
    with pytest.raises(RuntimeError, match="missing, duplicated, or unexpected bodies"):
        path_rows(paths, requested)
