"""Wall-time reconstruction must retain commands, holds and source provenance."""
import importlib.util
import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("playground_grasp", ROOT/"scripts/run_playground_grasp.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample_plan():
    return dict(schema_version=1, fps=2, expected_initial_state=[0.]*7,
                actions=[[float(i)]*7 for i in range(4)],
                phases=["open", "pregrasp", "close", "lift_hold"], source={"kind": "test"})


def write_log(path, times, action_indices, commands):
    rows = [dict(unix_s=time, action_index=index, applied=[float(command)]*7,
                 observed=[999.]*7, target=[888.]*7)
            for time, index, command in zip(times, action_indices, commands)]
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    return path


def test_applied_commands_offsets_and_gap(tmp_path):
    first = write_log(tmp_path/"first.jsonl", [10., 11.], [0, 1], [0, 1])
    second = write_log(tmp_path/"second.jsonl", [14., 15.], [0, 1], [2, 3])
    result = module.retime_applied_commands(sample_plan(), [first, second], [0, 2])
    assert np.asarray(result["actions"])[:, 0].tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 3]
    assert result["phases"] == ["open"]*2+["pregrasp"]*6+["close"]*2+["lift_hold"]
    assert result["command_provenance"][6]["applied_unix_s"] == 11.
    assert result["command_provenance"][-1]["source_index"] == 1
    assert result["waypoints"][-1]["end_frame"] == 10


def test_video_crop_still_simulates_physics_prefix(tmp_path):
    log = write_log(tmp_path/"commands.jsonl", [10., 11.], [0, 3], [0, 3])
    result = module.retime_applied_commands(sample_plan(), [log], [0], start_unix=10.5, end_unix=12.)
    assert result["timing"]["simulation_start_unix_s"] == 10.
    assert result["timing"]["video_start_frame"] == 1
    assert len(result["actions"]) == 4
    assert result["expected_initial_state"] == [0.]*7


def test_preroll_holds_reviewed_initial_state(tmp_path):
    log = write_log(tmp_path/"commands.jsonl", [10.], [3], [3])
    result = module.retime_applied_commands(sample_plan(), [log], [0], start_unix=9.)
    assert result["actions"][:2] == [[0.]*7]*2
    assert result["phases"][:2] == ["initial_hold"]*2
    assert result["command_provenance"][:2] == [None, None]


def test_rejects_overlapping_or_nonchronological_logs(tmp_path):
    first = write_log(tmp_path/"first.jsonl", [10., 11.], [0, 1], [0, 1])
    second = write_log(tmp_path/"second.jsonl", [10.5], [0], [2])
    with pytest.raises(ValueError, match="Non-increasing"):
        module.retime_applied_commands(sample_plan(), [first, second], [0, 2])


def test_missing_applied_never_falls_back_to_observed_or_target(tmp_path):
    log = tmp_path/"commands.jsonl"
    log.write_text(json.dumps(dict(unix_s=10., action_index=3, observed=[1.]*7, target=[2.]*7))+"\n")
    with pytest.raises(ValueError, match="applied command"):
        module.retime_applied_commands(sample_plan(), [log], [0])


def test_offsets_must_correspond_to_original_plan(tmp_path):
    log = write_log(tmp_path/"commands.jsonl", [10.], [3], [3])
    with pytest.raises(ValueError, match="original action index"):
        module.retime_applied_commands(sample_plan(), [log], [1])
    with pytest.raises(ValueError, match="one action offset"):
        module.retime_applied_commands(sample_plan(), [log], [])


def test_observation_audit_never_mutates_drive_commands(tmp_path):
    log = write_log(tmp_path/"commands.jsonl", [10., 11.], [0, 3], [0, 3])
    plan = module.retime_applied_commands(sample_plan(), [log], [0])
    before = np.asarray(plan["actions"]).copy()
    audited = module.observation_audit_samples(plan)
    assert all(np.array_equal(row, np.full(7, 999.)) for row in audited)
    assert np.array_equal(np.asarray(plan["actions"]), before)


def test_stopped_plan_requires_explicit_incomplete_mode():
    cfg = json.loads((ROOT/"configs/piper_playground.json").read_text())
    initial = cfg["initial_state_joint_rad_m"]
    plan = dict(schema_version=1, fps=cfg["fps"], actions=[initial],
                expected_initial_state=initial, phases=["close"])
    with pytest.raises(ValueError, match="lift_hold"):
        module.validate_input_plan(plan, cfg)
    assert module.validate_input_plan(plan, cfg, allow_incomplete=True)["phases"] == ["close"]


@pytest.mark.parametrize("lift,complete,expected", [(0., True, False), (.05, True, True), (.05, False, None)])
def test_physics_outcome_is_json_serializable_without_isaac(lift, complete, expected):
    # Run the outcome expression with the NumPy values returned by PhysX. A
    # failed short-circuit used to retain numpy.bool_, losing the JSON report.
    tree = ast.parse(inspect.getsource(module.simulate))
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "success" for target in node.targets))
    expression = compile(ast.Expression(assignment.value), "physics outcome", "eval")
    result = eval(expression, {}, dict(held_height=[np.float64(lift)], stable=True,
        contacts_at_hold=True, bilateral_hold_fraction=1., return_distance=np.float64(0.),
        unintended=[], complete=complete))
    assert result is expected
    assert json.loads(json.dumps({"success": result}))["success"] is expected
