"""Table-relative TCP goals must remain correct when the robot base is raised."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("playground_grasp_heights", ROOT/"scripts/run_playground_grasp.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def config():
    cfg = json.loads((ROOT/"configs/piper_playground.json").read_text())
    cfg["objects"] = copy.deepcopy(cfg["calibration_objects"])
    cfg["robot_base_height_above_table_m"] = .014
    return cfg


def waypoint(plan, name):
    return next(row for row in plan["waypoints"] if row["name"] == name)


def test_default_with_raised_base_targets_twenty_mm_above_table():
    cfg = config()
    plan = module.make_plan(cfg)
    assert plan["grasp_height_m"] == pytest.approx(.006)
    assert waypoint(plan, "grasp_open")["tcp_base_m"][2]+.014 == pytest.approx(.020, abs=1e-5)
    assert waypoint(plan, "pregrasp")["tcp_base_m"][2]+.014 == pytest.approx(.080, abs=1e-5)
    assert plan["source"]["height_reference"]["input_reference"] == "tabletop_z"
    cfg["robot_base_height_above_table_m"] = .100
    assert plan["source"]["configuration"]["robot_base_height_above_table_m"] == .014


def test_twelve_mm_goal_allows_negative_base_z_and_records_geometry_separately():
    cfg = config()
    green = next(obj for obj in cfg["objects"] if obj["name"] == "GreenCube")
    green["size"] = [.029]*3
    green["clearance"] = .003
    plan = module.make_plan(cfg, grasp_height_above_table=.012)
    assert plan["grasp_height_base_m"] == pytest.approx(-.002)
    assert plan["min_planned_tcp_z_m"] == pytest.approx(-.002, abs=1e-5)
    assert waypoint(plan, "grasp_open")["tcp_base_m"][2]+.014 == pytest.approx(.012, abs=1e-5)
    geometry = plan["grasp_geometry"]
    assert geometry["object_spawn_clearance_m"] == .003
    assert geometry["nominal_settled_object_top_above_table_m"] == .029
    assert geometry["nominal_tcp_depth_below_settled_object_top_m"] == pytest.approx(.017)
    assert "not a finger-body clearance" in geometry["interpretation"]
    # The change describes TCP/reference frames, not a rewrite of the scene or cube.
    assert cfg["objects"] == plan["source"]["configuration"]["objects"]


def test_explicit_legacy_option_retains_base_frame_pregrasp_and_grasp():
    plan = module.make_plan(config(), grasp_height=.006)
    assert waypoint(plan, "pregrasp")["tcp_base_m"][2] == pytest.approx(.080, abs=1e-5)
    assert waypoint(plan, "grasp_open")["tcp_base_m"][2] == pytest.approx(.006, abs=1e-5)
    assert plan["grasp_height_above_table_m"] == pytest.approx(.020)
    assert plan["input_reference"] == "legacy_robot_base_z"


def test_zero_mount_default_preserves_old_targets_and_timing():
    cfg = config()
    cfg["robot_base_height_above_table_m"] = 0.
    new = module.make_plan(cfg)
    legacy = module.make_plan(cfg, grasp_height=.020)
    np.testing.assert_array_equal(new["actions"], legacy["actions"])
    assert new["phases"] == legacy["phases"]


@pytest.mark.parametrize("mount", [-.001, float("nan"), float("inf")])
def test_rejects_invalid_mount_before_ik(mount):
    with pytest.raises(ValueError, match="robot_base_height_above_table_m"):
        module.make_plan({"robot_base_height_above_table_m": mount})


@pytest.mark.parametrize("height", [-.002, 0., .080, .100, float("nan"), float("inf")])
def test_rejects_invalid_tabletop_goal_before_ik(height):
    with pytest.raises(ValueError, match="Grasp"):
        module.make_plan({}, grasp_height_above_table=height)


def test_rejects_ambiguous_height_reference():
    with pytest.raises(ValueError, match="either"):
        module.make_plan({}, grasp_height=.012, grasp_height_above_table=.012)


@pytest.mark.parametrize("width", [.019, .029, float("nan"), float("inf")])
def test_rejects_invalid_close_width_before_ik(width):
    with pytest.raises(ValueError, match="Close width"):
        module.make_plan({}, close_width=width)


def test_existing_plan_cli_keeps_trajectory_and_height_provenance(tmp_path):
    cfg = config()
    original = module.make_plan(cfg, grasp_height_above_table=.012)
    # A different scene mount must not cause new planner defaults to rewrite a
    # recorded joint sequence supplied via --plan-input.
    cfg["robot_base_height_above_table_m"] = .022
    cfg_path, plan_path, output = tmp_path/"config.json", tmp_path/"plan.json", tmp_path/"rerun"
    cfg_path.write_text(json.dumps(cfg))
    plan_path.write_text(json.dumps(original))
    result = subprocess.run([sys.executable, str(ROOT/"scripts/run_playground_grasp.py"),
        "--config", str(cfg_path), "--plan-input", str(plan_path), "--output", str(output), "--plan-only"],
        text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((output/"plan.json").read_text()) == original


def test_cli_rejects_height_override_of_existing_plan(tmp_path):
    cfg_path = tmp_path/"config.json"
    cfg_path.write_text(json.dumps(config()))
    result = subprocess.run([sys.executable, str(ROOT/"scripts/run_playground_grasp.py"),
        "--config", str(cfg_path), "--plan-input", str(tmp_path/"missing.json"),
        "--output", str(tmp_path/"rerun"), "--plan-only", "--grasp-height-above-table", ".012"],
        text=True, capture_output=True)
    assert result.returncode == 2
    assert "preserves recorded actions" in result.stderr
