"""Real-URDF workspace checks beyond the original small XY rectangle."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pressb.kinematics import PiperKinematics
from pressb.table_buttons.workspace import PressWorkspace


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def workspace():
    cfg = json.loads((ROOT / "configs/piper_table_buttons.json").read_text())
    # Two useful tilts suffice to test both radial envelopes and the lower
    # hover fallback. All points still use actual URDF IK, not a fake solver.
    cfg["press_workspace"] = dict(pitch_candidates_deg=[160, 150],
                                  hover_z_candidates_m=[.12, .08],
                                  radial_bounds_m=[.08, .62], radial_step_m=.02)
    kin = PiperKinematics(ROOT / cfg["robot_urdf"], tip_offset=.1358)
    return PressWorkspace(kin, cfg)


def check_approach(workspace, xy, approach):
    assert approach is not None
    assert approach["vertical_sample_count"] >= 9
    np.testing.assert_allclose(approach["path_xyz"][:, :2], np.tile(xy, (len(approach["q_path"]), 1)))
    assert np.all(np.diff(approach["path_xyz"][:, 2]) < 0.)
    assert np.min(np.abs(approach["path_xyz"][:, 2]-.043)) < 1e-12
    for q, target in zip(approach["q_path"], approach["path_xyz"]):
        actual = workspace.kin.fk(q)
        np.testing.assert_allclose(actual[:3, 3], target, atol=5e-4)
        assert Rotation.from_matrix(approach["orientation"] @ actual[:3, :3].T).magnitude() < 1e-2
        assert np.all(q >= workspace.kin.lower) and np.all(q <= workspace.kin.upper)
    assert np.max(np.abs(np.diff(approach["q_path"], axis=0))) <= workspace.max_joint_step_rad


def test_default_workspace_region_prefers_existing_pitch_and_hover(workspace):
    result = workspace.solve([.33, .12])
    check_approach(workspace, [.33, .12], result)
    assert result["pitch_deg"] == 160.
    assert result["hover_z_m"] == .12


def test_alternate_pitch_reaches_outer_region_beyond_old_rectangle(workspace):
    result = workspace.solve([.52, .18])
    check_approach(workspace, [.52, .18], result)
    assert result["pitch_deg"] == 150.
    assert result["hover_z_m"] == .12


def test_lower_hover_reaches_near_radius_where_high_hover_is_infeasible(workspace):
    result = workspace.solve([.10, 0.])
    check_approach(workspace, [.10, 0.], result)
    assert result["hover_z_m"] == .08


def test_solutions_exposes_alternatives_for_the_collision_checker(workspace):
    solutions = list(workspace.solutions([.35, .1]))
    assert {(s["pitch_deg"], s["hover_z_m"]) for s in solutions} == {
        (160., .12), (150., .12), (160., .08), (150., .08),
    }
    for solution in solutions:
        check_approach(workspace, [.35, .1], solution)


def test_yaw_rotates_first_joint_and_reuses_exact_radial_solution(workspace):
    radius, theta = .36, -.7
    radial = workspace.solve([radius, 0.])
    xy = radius*np.array([np.cos(theta), np.sin(theta)])
    rotated = workspace.solve(xy)
    check_approach(workspace, xy, rotated)
    np.testing.assert_allclose(rotated["q_path"][:, 1:], radial["q_path"][:, 1:], atol=1e-12)
    np.testing.assert_allclose(rotated["q_path"][:, 0] - radial["q_path"][:, 0], theta, atol=1e-12)


def test_base_joint_limit_is_enforced_even_when_radius_is_reachable(workspace):
    theta = workspace.kin.upper[0] + .1
    assert workspace.solve(.35*np.array([np.cos(theta), np.sin(theta)])) is None


def test_outside_urdf_scan_radius_is_not_returned_as_a_feasible_approach(workspace):
    assert workspace.solve([1., 0.]) is None
    assert workspace.solve([0., 0.]) is None


def test_profiles_are_serializable_and_do_not_need_repeated_ik(workspace, monkeypatch):
    encoded = json.dumps(workspace.radial_profiles)
    assert "intervals_m" in encoded and "samples" in encoded
    radial = .36
    workspace.solve([radial, 0.])
    def no_solver(*args, **kwargs):
        raise AssertionError("An already solved radius was needlessly recomputed")
    monkeypatch.setattr(workspace.kin, "solve", no_solver)
    check_approach(workspace, [radial, 0.], workspace.solve([radial, 0.]))
    assert json.dumps(workspace.radial_profiles) == encoded


def test_radial_shortcut_rejects_a_non_z_first_joint(workspace):
    kin = deepcopy(workspace.kin)
    for index, (origin, axis) in enumerate(kin._chain):
        if axis is not None:
            kin._chain[index] = (origin, np.array([1., 0., 0.]))
            break
    with pytest.raises(ValueError, match="first joint"):
        PressWorkspace(kin, dict(initial_joint_positions_rad=workspace.initial))


def test_discontinuous_branch_is_rejected_even_if_each_pose_has_an_ik_solution(workspace, monkeypatch):
    local = PressWorkspace(workspace.kin, dict(initial_joint_positions_rad=workspace.initial,
                                             press_workspace=dict(max_joint_step_rad=.001)))
    # The true arm changes by more than 1 mrad over the vertical samples. This
    # demonstrates that accepting only isolated successful IKs is insufficient.
    assert local._radial_path(.35, 160., .12, local.initial) is None
