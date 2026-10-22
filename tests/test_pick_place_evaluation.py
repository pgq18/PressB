"""Geometry, paired experimental design and failure-denominator checks."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pressb.pick_place.evaluation import (COLORS, effective_config, make_plan, placement_region,
                                        read_reports, sample_polygon, summarize_results, wilson_interval,
                                        write_summary)


@pytest.fixture
def config():
    return json.loads((Path(__file__).resolve().parents[1]/"configs/piper_pick_place.json").read_text())


def test_plan_is_reproducible_paired_balanced_and_footprints_inside_markers(config):
    plan = make_plan(config)
    assert plan == make_plan(config)
    assert plan != make_plan(config, seed=123)
    red, color = plan["episodes"][:100], plan["episodes"][100:]
    assert [e["center_xy"] for e in red] == [e["center_xy"] for e in color]
    assert all(e["color"] == "red" for e in red)
    assert all(sum(e["color"] == name for e in color) == 20 for name in COLORS)
    assert sum(e["record_video"] for e in plan["episodes"]) == 7
    halfplanes = np.asarray(plan["region"]["marker_halfplanes"])
    offsets = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])*.015
    for episode in red:
        corners = np.asarray(episode["center_xy"])+offsets
        assert np.all(corners @ halfplanes[:, :2].T+halfplanes[:, 2] <= 1e-12)
    assert plan["version"] == 2
    assert plan["region"]["valid_center_area_m2"] == pytest.approx(.04638627373080603)


def test_real_markers_define_edges_not_vertices_and_exact_rectangle_bounds(config):
    region = placement_region(config)
    boundary = np.asarray(region["marker_polygon_xy"])
    np.testing.assert_allclose(boundary.min(axis=0), [.2269872354916476, -.1463872676459559])
    np.testing.assert_allclose(boundary.max(axis=0), [.49717463437046094, .07673807509615128])
    np.testing.assert_allclose(region["valid_center_bounds_xy_m"],
                               [[.2419872354916476, -.1313872676459559],
                                [.48217463437046094, .06173807509615128]])
    assert region["marker_polygon_area_m2"] == pytest.approx(.06028565597943365)
    halfplanes = np.asarray(region["marker_halfplanes"])
    for point, edge in zip(region["marker_centres_xy"], region["edge_lines"]):
        distances = halfplanes[:, :2] @ point + halfplanes[:, 2]
        assert np.sum(np.isclose(distances, 0., atol=1e-12)) == 1
        assert np.all(distances <= 1e-12)
        assert np.min(np.linalg.norm(boundary-point, axis=1)) > .01
        assert np.dot(edge["outward_normal_xy"], edge["long_axis_xy"]) == pytest.approx(0., abs=1e-12)
    sides = np.roll(boundary, -1, axis=0)-boundary
    np.testing.assert_allclose(sides[0], -sides[2], atol=1e-12)
    np.testing.assert_allclose(sides[1], -sides[3], atol=1e-12)
    assert np.dot(sides[0], sides[1]) == pytest.approx(0., abs=1e-12)


def test_rectangle_with_rotated_global_frame_and_independently_rotated_cube(config):
    angle = np.deg2rad(27.)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shift = np.array([.7, -.4])
    original = placement_region(config, yaw_deg=35.)
    transformed = copy.deepcopy(config)
    for marker in transformed["tape_markers"]:
        marker["xy"] = (rotation @ marker["xy"] + shift).tolist()
        marker["yaw_deg"] += 27.
    # Marker order and opposite long-axis sign must not affect the domain.
    transformed["tape_markers"].reverse()
    transformed["tape_markers"][1]["yaw_deg"] += 180.
    actual = placement_region(transformed, yaw_deg=62.)
    expected = np.asarray(original["valid_center_polygon_xy"]) @ rotation.T + shift
    for vertex in actual["valid_center_polygon_xy"]:
        assert np.min(np.linalg.norm(expected-vertex, axis=1)) < 1e-12
    assert actual["valid_center_area_m2"] == pytest.approx(original["valid_center_area_m2"])


@pytest.mark.parametrize("change", ["missing_yaw", "not_rectangle", "coincident_edges", "outside_segment", "too_large_cube"])
def test_invalid_marker_edge_geometry_does_not_fall_back_to_hull(config, change):
    if change == "missing_yaw":
        del config["tape_markers"][0]["yaw_deg"]
    elif change == "not_rectangle":
        config["tape_markers"][0]["yaw_deg"] = 80.
    elif change == "coincident_edges":
        config["tape_markers"][3]["xy"][0] = config["tape_markers"][0]["xy"][0]
    elif change == "outside_segment":
        config["tape_markers"][0]["xy"][1] = 10.
    with pytest.raises(ValueError):
        placement_region(config, edge=1. if change == "too_large_cube" else .03)


def test_eroded_vertices_support_cube_at_nonzero_yaw(config):
    region = placement_region(config, yaw_deg=35)
    theta = np.deg2rad(35)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    offsets = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])*.015 @ rotation.T
    halfplanes = np.asarray(region["marker_halfplanes"])
    for center in region["valid_center_polygon_xy"]:
        distances = (center+offsets) @ halfplanes[:, :2].T+halfplanes[:, 2]
        assert distances.max() <= 1e-10
        assert np.abs(distances).min() < 1e-10


def test_uniform_sampling_uses_triangle_area_not_equal_triangle_probability():
    polygon = [[0, 0], [4, 0], [4, 1], [0, 3]]
    points = sample_polygon(polygon, 100000, np.random.default_rng(70))
    # Fan triangles have areas 2 and 6; first triangle is below y=x/4.
    assert np.mean(points[:, 1] <= points[:, 0]/4) == pytest.approx(.25, abs=.006)


def test_effective_config_only_changes_smoothing_and_does_not_mutate(config):
    original = copy.deepcopy(config)
    changed = effective_config(config)
    assert changed["policy_execution"]["smoothing"] == dict(mode="joint_ema", alpha=.8)
    changed["policy_execution"] = original["policy_execution"]
    assert changed == original == config


@pytest.mark.parametrize("section,key,value", [
    ("cube", "edge", .04),
    ("cube", "mass", .05),
    ("cube", "yaw_deg", 35.),
    (None, "physics_dt", 1/60),
    (None, "fps", 15),
    ("episode_source", "episode_index", 461),
    ("episode_source", "trajectory", "other/episode_000461.npz"),
])
def test_fixed_evaluation_preset_rejects_different_physical_protocol(config, section, key, value):
    target = config if section is None else config.setdefault(section, {})
    target[key] = value
    with pytest.raises(ValueError, match="PAP evaluation preset requires"):
        make_plan(config)


def test_episode461_configuration_cannot_claim_default_cube_evaluation():
    path = Path(__file__).resolve().parents[1]/"configs/piper_pick_place_episode_461.json"
    with pytest.raises(ValueError, match="PAP evaluation preset requires cube edge"):
        make_plan(json.loads(path.read_text()))


def test_reported_timeout_follows_explicit_plan_duration(config):
    plan = make_plan(config, episodes_per_group=5, duration_s=12.)
    assert plan["duration_s"] == 12.
    assert plan["protocol"]["timeout_rule"] == "No physical success by 12 simulated seconds is failure"


def test_failure_denominator_includes_timeout_invalid_model_and_excludes_infrastructure(config):
    plan = make_plan(config, episodes_per_group=5)
    reports = {
        "red_000": dict(physical_success=True, evaluation_status="complete", physical_outcome=dict(first_success_time_s=7.2)),
        "red_001": dict(physical_success=False, evaluation_status="complete", failure_reason="timeout"),
        "red_002": dict(physical_success=False, evaluation_status="complete", failure_reason="invalid_policy_action", evaluation_error="ValueError"),
        "red_003": dict(physical_success=False, evaluation_status="interrupted", evaluation_error="Socket closed"),
    }
    summary = summarize_results(plan, reports)
    stats = summary["groups"]["position_red"]
    assert not summary["complete"] and summary["completed"] == 3
    assert stats["successes"] == 1 and stats["failures"] == 2
    assert stats["success_rate"] == pytest.approx(1/3)
    assert stats["infrastructure_errors"] == 1 and stats["pending"] == 1
    assert stats["mean_first_success_time_s"] == 7.2
    assert stats["failure_reasons"] == dict(timeout=1, invalid_policy_action=1)
    assert len(summary["failures"]) == 2


def test_completed_evaluation_requires_all_planned_reports(config):
    plan = make_plan(config, episodes_per_group=5)
    reports = {episode["episode_id"]: dict(physical_success=False) for episode in plan["episodes"]}
    summary = summarize_results(plan, reports)
    assert summary["complete"] and summary["completed"] == 10
    assert summary["groups"]["position_red"]["success_rate"] == 0
    assert wilson_interval(0, 0) is None
    assert wilson_interval(0, 100)[1] == pytest.approx(.036993498)
    assert wilson_interval(100, 100)[0] == pytest.approx(.963006502)


def test_legacy_wrong_domain_notice_preserves_outcomes_and_plan(config, tmp_path):
    plan = make_plan(config, episodes_per_group=5)
    reports = {episode["episode_id"]: dict(physical_success=episode["pair_index"] % 2 == 0)
               for episode in plan["episodes"]}
    current = summarize_results(plan, reports)
    assert current["sampling_region_matches_request"]
    assert current["sampling_region_notice"] is None
    plan["version"] = 1
    del plan["region"]["region_kind"]
    legacy = summarize_results(plan, reports)
    assert not legacy["sampling_region_matches_request"]
    assert "Incorrect sampling region" in legacy["sampling_region_notice"]
    assert "must not be reported" in legacy["sampling_region_notice"]
    assert legacy["groups"] == current["groups"]
    assert legacy["episodes"] == current["episodes"]
    assert legacy["complete"] == current["complete"]
    plan_text = json.dumps(plan)
    (tmp_path/"plan.json").write_text(plan_text)
    (tmp_path/"results.jsonl").write_text("\n".join(json.dumps(dict(episode_id=key, **report))
                                                   for key, report in reports.items()))
    write_summary(tmp_path)
    assert "**Sampling region correction:**" in (tmp_path/"summary.md").read_text()
    assert (tmp_path/"plan.json").read_text() == plan_text


def test_reports_override_jsonl_and_live_incomplete_line_is_ignored(tmp_path):
    run = tmp_path/"run"
    report_dir = run/"episodes"/"red_000"
    report_dir.mkdir(parents=True)
    (run/"results.jsonl").write_text('{"episode":{"episode_id":"red_000"},"physical_success":false}\n{"episode":')
    (report_dir/"report.json").write_text('{"physical_success":true}')
    assert read_reports(tmp_path)["red_000"]["physical_success"]


@pytest.mark.parametrize("count", [0, -5, 3, True])
def test_invalid_group_counts_rejected(config, count):
    with pytest.raises(ValueError):
        make_plan(config, episodes_per_group=count)
