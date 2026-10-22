"""Exact candidate checks, whole camera-domain sampling, and layout collisions."""
from copy import deepcopy

import numpy as np
import pytest

from pressb.table_buttons import placement
from pressb.table_buttons.task import COLORS


def approach(xy, branch=0.):
    q = np.r_[xy, branch, 0., 0., 0.]
    q_press = q + [0., 0., .1, 0., 0., 0.]
    return dict(press=np.r_[xy, .019], q_hover=q, q_press=q_press,
                q_path=np.array([q, q_press]), pitch_deg=160.+branch)


@pytest.fixture
def harness(monkeypatch):
    camera_pose = np.eye(4)
    camera_pose[2, 3] = 1.
    cfg = dict(
        initial_joint_positions_rad=[0.]*6, initial_gripper_width_m=.0002,
        up_camera=dict(base_from_camera_usd=camera_pose.tolist(), resolution=[640, 480],
                       fx=500., fy=500., cx=320., cy=240.),
        button_geometry=dict(base_radius_m=.022, base_height_m=.017, cap_height_m=.008),
        position_sampling=dict(bounds_xy=[[.30, .39], [-.14, .14]], min_separation=.10,
                               pixel_margin=16., max_attempts=500),
    )
    domain = dict(polygon_xy_m=[[-.4, -.3], [.4, -.3], [-.4, .3]],
                  bounds_xy_m=[[-.4, .4], [-.3, .3]], area_m2=.24)

    class Workspace:
        def __init__(self):
            self.calls = []
            self.propose = lambda xy: [approach(xy)]

        def solutions(self, xy):
            self.calls.append(np.asarray(xy).copy())
            yield from self.propose(xy)

    class Clearance:
        def __init__(self):
            self.calls, self.button_calls = [], []
            self.accept = lambda path: True
            self.accept_buttons = lambda path, positions, target: True

        def check_path(self, path, width):
            self.calls.append((np.asarray(path).copy(), width))
            ok = self.accept(path)
            return ok, {"reason": "clear" if ok else "table_collision"}

        def check_button_path(self, path, positions, target, width):
            self.button_calls.append((np.asarray(path).copy(), deepcopy(positions), target, width))
            ok = self.accept_buttons(path, positions, target)
            return ok, {"reason": "clear" if ok else "button_collision"}

    workspace, clearance = Workspace(), Clearance()
    monkeypatch.setattr(placement, "camera_center_domain", lambda cfg: deepcopy(domain))
    monkeypatch.setattr(placement, "PressWorkspace", lambda kin, cfg: workspace)
    monkeypatch.setattr(placement, "RobotClearance", lambda kin, cfg: clearance)
    return cfg, domain, workspace, clearance


def test_polygon_rejects_invisible_envelope_corner_before_ik(harness):
    cfg, _, workspace, clearance = harness
    sampler = placement.ReachablePlacement(None, cfg)
    assert sampler.candidate([.35, .25]) is None  # Inside bounds, outside triangle.
    assert sampler.last_diagnostic["reason"] == "outside_camera_domain"
    assert workspace.calls == clearance.calls == []
    assert sampler.candidate([-.25, .1]) is not None


def test_full_housing_projection_is_rechecked_even_inside_polygon(harness):
    cfg, domain, workspace, _ = harness
    # A stale/overly broad domain must not bypass the actual camera predicate.
    domain["polygon_xy_m"] = [[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]
    domain["bounds_xy_m"] = [[-1., 1.], [-1., 1.]]
    sampler = placement.ReachablePlacement(None, cfg)
    assert sampler.candidate([.9, 0.]) is None
    assert workspace.calls == []


def test_collision_on_transit_rejects_clear_endpoints(harness):
    cfg, _, _, clearance = harness
    # The endpoint has x=.3, but the transit passes through the obstacle.
    clearance.accept = lambda path: not np.any((path[:, 0] > .13) & (path[:, 0] < .17))
    sampler = placement.ReachablePlacement(None, cfg)
    assert sampler.candidate([.3, -.25]) is None
    path, width = clearance.calls[0]
    np.testing.assert_array_equal(path[0], cfg["initial_joint_positions_rad"])
    assert np.max(np.abs(np.diff(path, axis=0))) <= .035+1e-12
    assert width == cfg["initial_gripper_width_m"]
    assert sampler.last_diagnostic["reason"] == "path_collision"


@pytest.mark.parametrize("rejection", ["table", "target_housing"])
def test_collision_tries_another_ik_orientation(harness, rejection):
    cfg, _, workspace, clearance = harness
    workspace.propose = lambda xy: [approach(xy, .2), approach(xy, .7)]
    if rejection == "table":
        clearance.accept = lambda path: path[-1, 2] > .5
    else:
        clearance.accept_buttons = lambda path, positions, target: path[-1, 2] > .5
    sampler = placement.ReachablePlacement(None, cfg)
    selected = sampler.candidate([-.2, 0.])
    assert selected["pitch_deg"] == 160.7
    assert sampler.last_diagnostic["rejected_approaches"] == 1
    assert len(clearance.calls) == 2
    assert clearance.button_calls[-1][1] == {"red": [-.2, 0.]}


def test_candidate_uses_requested_coordinates_without_grid_rounding(harness):
    cfg, _, workspace, _ = harness
    xy = np.array([-.23456789123, .01234567891])
    workspace.propose = lambda point: [approach(point)] if np.array_equal(point, xy) else []
    sampler = placement.ReachablePlacement(None, cfg)
    assert sampler.candidate(xy) is not None
    assert sampler.candidate(xy + [1e-5, 0.]) is None
    np.testing.assert_array_equal(workspace.calls[0], xy)
    assert sampler.last_diagnostic["reason"] == "no_ik"


def test_seeded_sampling_uses_complete_domain_and_checks_every_layout_path(harness):
    cfg, _, _, clearance = harness
    original = deepcopy(cfg)
    sampler = placement.ReachablePlacement(None, cfg)
    first, same, different = sampler.sample(12), sampler.sample(12), sampler.sample(13)
    assert first["positions"] == same["positions"]
    assert first["positions"] != different["positions"]
    assert cfg == original
    points = np.asarray(list(first["positions"].values()))
    assert set(first["positions"]) == set(first["approaches"]) == set(COLORS)
    assert np.any(points[:, 0] < .30)  # Explicit legacy bounds are ignored.
    for index, point in enumerate(points):
        assert sampler._visible(point)
        for other in points[index+1:]:
            assert np.linalg.norm(point-other) >= .10
    complete_layout_calls = [row for row in clearance.button_calls if len(row[1]) == 3]
    assert [row[2] for row in complete_layout_calls] == list(COLORS)*3
    assert first["domain"]["sampling"]["complete_button_layout_check"]
    assert first["domain"]["sampling"]["exact_candidate_ik"]


def test_layout_collision_restarts_sampling_instead_of_returning_unsafe_positions(harness):
    cfg, _, _, clearance = harness
    rejected_once = []

    def accept(path, positions, target):
        if len(positions) == 3 and not rejected_once:
            rejected_once.append(deepcopy(positions))
            return False
        return True

    clearance.accept_buttons = accept
    sampled = placement.ReachablePlacement(None, cfg).sample(12)
    assert sampled["positions"] != rejected_once[0]
    assert sampled["domain"]["sampling"]["rejected_by_reason"]["button_path_collision"] == 1


def test_configured_valid_separation_below_default_is_respected(harness):
    cfg, _, _, _ = harness
    cfg["position_sampling"]["min_separation"] = .05
    sampler = placement.ReachablePlacement(None, cfg)
    assert sampler.min_separation == .05
    assert sampler.sample(8)["domain"]["sampling"]["min_separation_m"] == .05


def test_grid_reports_polygon_and_exact_reachability_separately(harness):
    cfg, _, workspace, clearance = harness
    workspace.propose = lambda xy: [approach(xy)] if xy[0] < -.1 else []
    clearance.accept = lambda path: path[-1, 1] < .05
    sampler = placement.ReachablePlacement(None, cfg)
    grid = sampler.grid(.1)
    points = grid["points_xy"]
    assert points.shape[1] == 2
    assert len(points) == len(grid["visible_mask"]) == len(grid["reachable_mask"])
    assert np.all(~grid["reachable_mask"] | grid["visible_mask"])
    expected = grid["visible_mask"] & (points[:, 0] < -.1) & (points[:, 1] < .05)
    np.testing.assert_array_equal(grid["reachable_mask"], expected)
    assert grid["resolution_m"] == .1
    assert grid["bounds_xy_m"] == [[-.4, .4], [-.3, .3]]
    assert len(workspace.calls) == np.sum(grid["visible_mask"])
    assert {reason for reason in grid["reasons"]} == {"clear", "no_ik", "path_collision", "outside_camera_domain"}
    assert [value is not None for value in grid["approaches"]] == expected.tolist()


def test_empty_domain_has_empty_grid_and_sampling_failure(harness):
    cfg, domain, workspace, _ = harness
    domain.update(polygon_xy_m=[], bounds_xy_m=None, area_m2=0.)
    sampler = placement.ReachablePlacement(None, cfg)
    assert sampler.grid()["points_xy"].shape == (0, 2)
    assert sampler.candidate([0., 0.]) is None
    with pytest.raises(ValueError, match="domain is empty"):
        sampler.sample(0)
    assert workspace.calls == []


def test_sampling_is_bounded_when_no_exact_candidate_has_ik(harness):
    cfg, _, workspace, _ = harness
    cfg["position_sampling"]["max_attempts"] = 12
    workspace.propose = lambda xy: []
    with pytest.raises(ValueError, match="after 12 proposals"):
        placement.ReachablePlacement(None, cfg).sample(0)
    assert 0 < len(workspace.calls) <= 12


def test_validate_layout_rejects_mismatched_approach_and_bad_spacing(harness):
    cfg, _, _, _ = harness
    sampler = placement.ReachablePlacement(None, cfg)
    positions = {"red": [-.3, -.2], "yellow": [-.1, -.2], "blue": [.1, -.2]}
    approaches = {color: approach(xy) for color, xy in positions.items()}
    assert sampler.validate_layout(positions, approaches)[0]
    approaches["red"] = approach([-.2, -.2])
    with pytest.raises(ValueError, match="different red button"):
        sampler.validate_layout(positions, approaches)
    positions["red"] = [-.15, -.2]
    approaches["red"] = approach(positions["red"])
    ok, diagnostic = sampler.validate_layout(positions, approaches)
    assert not ok and diagnostic["reason"] == "button_separation"


@pytest.mark.parametrize("xy", [[0.], [0., 0., 0.], [np.nan, 0.]])
def test_invalid_candidate_coordinates_raise(harness, xy):
    with pytest.raises(ValueError, match="two finite"):
        placement.ReachablePlacement(None, harness[0]).candidate(xy)


@pytest.mark.parametrize("spacing", [0., -1., np.inf, np.nan])
def test_invalid_grid_spacing_raises(harness, spacing):
    with pytest.raises(ValueError, match="spacing"):
        placement.ReachablePlacement(None, harness[0]).grid(spacing)


def test_overlapping_separation_configuration_is_rejected(harness):
    cfg = harness[0]
    cfg["position_sampling"]["min_separation"] = .04
    with pytest.raises(ValueError, match="disjoint"):
        placement.ReachablePlacement(None, cfg)
