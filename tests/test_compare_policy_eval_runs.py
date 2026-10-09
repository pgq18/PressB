"""Comparison must preserve condition identity and distinguish missing runs."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


SPEC = importlib.util.spec_from_file_location(
    "compare_policy_eval_runs", Path(__file__).resolve().parents[1] / "scripts/compare_policy_eval_runs.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def update(path, **values):
    record = json.loads(path.read_text())
    record.update(values)
    write(path, record)


def make_run(root, floors=(24, 25), shift=0, pixels=20):
    worker = root / "worker_0"
    config = {"physics_dt": 1 / 120, "robot_urdf": "vendor/robot.urdf"}
    smoothing = {"window": 3, "physics_hz": 120, "kind": "linear_joint_interpolation_then_causal_mean"}
    model_sha = "a" * 64
    schedule = [dict(episode_id=i, floor=floor, repeat=0, panel_layout_index=0,
                     panel_layout_name="center", panel_offset_x_m=0., panel_offset_y_m=0.,
                     seed_episode_index=floor - 24) for i, floor in enumerate(floors)]
    args = dict(expected_checkpoint_sha256=model_sha, expected_checkpoint_step=10600,
                smoothing_window=3, max_seconds=15., seed=20260930, gpu=0)
    manifest = dict(expected_checkpoint_sha256=model_sha, expected_checkpoint_step=10600,
                    scene_sha256="b" * 64, arguments=args, config=config, motion_smoothing=smoothing,
                    episode_schedule=schedule, panel_layout_mode="center_corners", sources={},
                    panel_layouts=[dict(panel_layout_name="center", panel_offset_x_m=0., panel_offset_y_m=0.)],
                    inference_seed_rule="base + index * 10000 + chunk", inference_clock="frozen")
    service = dict(status="ready", checkpoint_verified=True, model_sha256=model_sha, checkpoint_step=10600,
                   provenance=dict(model_sha256=model_sha, checkpoint="/different/machine/checkpoint"),
                   fps=30, action_horizon=7, camera_order=["global", "wrist"], resolution=224,
                   input_image_resize="RGB, bilinear square", num_inference_timesteps=4,
                   pose_frame="base_link", pose_link="gripper_tcp", xyz_units="metres", rotation6d="first two rows",
                   quaternion_order="wxyz", gripper_is_learned=False, controller_gripper_width_m=.008)
    rows = []
    for condition in schedule:
        i, floor = condition["episode_id"], condition["floor"]
        episode = worker / f"episode_{i:06d}"
        success = floor == 24
        row = dict(condition, task=f"Press {floor} floor.", success=success, task_success=success,
                   termination="target_pressed" if success else "time_limit",
                   sim_seconds=5. if success else 15., min_target_tip_distance_m=.001 if success else .1)
        rows.append(row)
        metadata = dict(row, config=config, motion_smoothing=smoothing, unexpected_collisions=[],
                        events=[dict(type="pressed", floor=floor)] if success else [])
        write(episode / "metadata.json", metadata)
        request = dict(chunk_index=0, observation_physics_index=0, observation_sim_time=0.,
                       task=row["task"], seed=20260930 + (floor - 24) * 10000,
                       state=[shift, 0, .2, 1, 0, 0, 0, .008], images={},
                       response=dict(actions_pose9=[[shift, 0, .2, 1, 0, 0, 0, 1, 0]] * 7))
        for camera in ("global", "wrist"):
            relative = Path("observations") / f"chunk_0000_{camera}.png"
            path = episode / relative
            path.parent.mkdir(exist_ok=True)
            image = Image.fromarray(np.full((12, 16, 3), pixels, np.uint8))
            image.save(path)
            request["images"][camera] = dict(path=str(relative), sha256=MODULE.sha(path),
                rgb_sha256=hashlib.sha256(np.asarray(image).tobytes()).hexdigest())
        write(episode / "requests.jsonl", request)
    write(worker / "eval_manifest.json", manifest)
    write(worker / "policy_service.json", service)
    write(worker / "status.json", dict(status="complete", completed_episodes=len(rows)))
    write(worker / "report.json", dict(complete=True, total_episodes=len(rows), passed_episodes=1,
        episodes=rows, fps=30, action_chunk_size=7, max_sim_seconds=15., motion_smoothing=smoothing))
    return root


def test_matches_conditions_after_episode_order_and_gpu_change(tmp_path):
    baseline = make_run(tmp_path / "baseline")
    candidate = make_run(tmp_path / "candidate", floors=(25, 24), shift=.003, pixels=30)
    path = candidate / "worker_0/eval_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["arguments"]["gpu"] = 1
    manifest["sources"] = {"different.py": {"sha256": "c" * 64}}
    write(path, manifest)
    result = MODULE.compare(baseline, candidate)
    assert result["comparison_valid"]
    assert result["baseline"]["episodes"] == 2
    assert result["agreement"]["termination_and_pressed_floors_match"] == 2
    assert result["episodes"][0]["candidate"]["episode_id"] == 1
    assert "execution_sources" in result["deployment_differences"][0]["differences"]
    assert result["metrics"]["state_position_error_m"]["mean"] == pytest.approx(.003)
    assert result["metrics"]["first_prediction_xyz_rmse_m"]["mean"] == pytest.approx(.003 / np.sqrt(3))
    for camera in ("global", "wrist"):
        assert result["image_metrics"][camera]["rgb_mae"]["mean"] == 10
        assert result["image_metrics"][camera]["rgb_rmse"]["mean"] == 10
        assert result["image_metrics"][camera]["resized_rgb_mae"]["mean"] == 10
    assert any(record["path"].endswith("requests.jsonl") for record in result["input_files"]["candidate"])
    assert "2/2" in MODULE.markdown(result)


@pytest.mark.parametrize("mutation", ["missing_episode", "failed_status", "missing_report_row", "wrong_offset", "duplicate_condition"])
def test_rejects_incomplete_or_mislabeled_conditions(tmp_path, mutation):
    baseline = make_run(tmp_path / "baseline")
    candidate = make_run(tmp_path / "candidate")
    worker = candidate / "worker_0"
    if mutation == "missing_episode":
        (worker / "episode_000001/metadata.json").unlink()
    elif mutation == "failed_status":
        update(worker / "status.json", status="failed")
    elif mutation == "missing_report_row":
        report = json.loads((worker / "report.json").read_text())
        report["episodes"].pop()
        write(worker / "report.json", report)
    elif mutation == "wrong_offset":
        update(worker / "episode_000001/metadata.json", panel_offset_x_m=.01)
    else:
        manifest = json.loads((worker / "eval_manifest.json").read_text())
        manifest["episode_schedule"].append(deepcopy(manifest["episode_schedule"][0]))
        write(worker / "eval_manifest.json", manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        MODULE.compare(baseline, candidate)


def test_rejects_different_complete_condition_coverage(tmp_path):
    baseline = make_run(tmp_path / "baseline")
    candidate = make_run(tmp_path / "candidate", floors=(24, 26))
    with pytest.raises(ValueError, match="coverage mismatch"):
        MODULE.compare(baseline, candidate)


def test_rejects_seed_difference_even_if_internal_request_seed_agrees(tmp_path):
    baseline = make_run(tmp_path / "baseline")
    candidate = make_run(tmp_path / "candidate")
    worker = candidate / "worker_0"
    manifest = json.loads((worker / "eval_manifest.json").read_text())
    manifest["arguments"]["seed"] += 1
    write(worker / "eval_manifest.json", manifest)
    for path in worker.glob("episode_*/requests.jsonl"):
        request = json.loads(path.read_text())
        request["seed"] += 1
        write(path, request)
    with pytest.raises(ValueError, match="identity mismatch: base_seed"):
        MODULE.compare(baseline, candidate)


def test_quaternion_sign_equivalence_and_missing_png_are_explicit(tmp_path):
    baseline = make_run(tmp_path / "baseline")
    candidate = make_run(tmp_path / "candidate")
    for path in candidate.glob("worker_*/episode_*/requests.jsonl"):
        request = json.loads(path.read_text())
        request["state"][3:7] = [-1, 0, 0, 0]
        write(path, request)
    for path in candidate.glob("worker_*/episode_*/observations/*global.png"):
        path.unlink()
    result = MODULE.compare(baseline, candidate)
    assert result["metrics"]["state_rotation_error_deg"]["max"] == 0
    assert result["image_metrics"]["global"]["rgb_mae"]["count"] == 0
    assert result["image_metrics"]["global"]["rgb_mae"]["mean"] is None
    assert result["image_metrics"]["wrist"]["rgb_mae"]["count"] == 2
    assert not result["episodes"][0]["initial_images"]["global"]["comparable"]


def test_detects_changed_lossless_input_file(tmp_path):
    baseline = make_run(tmp_path / "baseline")
    candidate = make_run(tmp_path / "candidate")
    Image.new("RGB", (16, 12)).save(candidate / "worker_0/episode_000000/observations/chunk_0000_global.png")
    with pytest.raises(ValueError, match="image file SHA mismatch"):
        MODULE.compare(baseline, candidate)


def test_failed_or_missing_run_cli_emits_invalid_report(tmp_path, monkeypatch):
    baseline = make_run(tmp_path / "baseline")
    output = tmp_path / "comparison"
    monkeypatch.setattr("sys.argv", ["compare", "--baseline-root", str(baseline),
                                    "--candidate-root", str(tmp_path / "missing"), "--output", str(output)])
    assert MODULE.main() == 2
    result = json.loads((output / "comparison.json").read_text())
    assert result["comparison_valid"] is False
    assert "timeouts" not in result
    assert "未计为超时" in (output / "summary.md").read_text()


@pytest.mark.parametrize("field", ["scene_sha256", "checkpoint", "config", "motion_smoothing"])
def test_rejects_different_evaluation_identity(tmp_path, field):
    baseline = make_run(tmp_path / "baseline")
    candidate = make_run(tmp_path / "candidate")
    worker = candidate / "worker_0"
    manifest = json.loads((worker / "eval_manifest.json").read_text())
    if field == "scene_sha256":
        manifest[field] = "d" * 64
    elif field == "checkpoint":
        manifest["expected_checkpoint_sha256"] = manifest["arguments"]["expected_checkpoint_sha256"] = "d" * 64
        service = json.loads((worker / "policy_service.json").read_text())
        service["model_sha256"] = service["provenance"]["model_sha256"] = "d" * 64
        write(worker / "policy_service.json", service)
    else:
        if field == "config":
            manifest[field]["table_height"] = .9
        else:
            manifest[field]["window"] = manifest["arguments"]["smoothing_window"] = 5
            update(worker / "report.json", motion_smoothing=manifest[field])
        for path in worker.glob("episode_*/metadata.json"):
            update(path, **{field: manifest[field]})
    write(worker / "eval_manifest.json", manifest)
    with pytest.raises(ValueError, match="identity mismatch"):
        MODULE.compare(baseline, candidate)


def test_failure_is_separate_from_timeout_and_error_cannot_hide_in_timeout(tmp_path):
    baseline = make_run(tmp_path / "baseline")
    candidate = make_run(tmp_path / "candidate")
    worker = candidate / "worker_0"
    path = worker / "episode_000001/metadata.json"
    update(path, error="inference returned invalid action")
    with pytest.raises(ValueError, match="execution error"):
        MODULE.compare(baseline, candidate)
    update(path, termination="invalid_policy_action")
    report = json.loads((worker / "report.json").read_text())
    report["episodes"][1]["termination"] = "invalid_policy_action"
    write(worker / "report.json", report)
    result = MODULE.compare(baseline, candidate)
    assert result["candidate"]["invalid_policy_actions"] == 1
    assert result["candidate"]["timeouts"] == 0


def test_records_candidate_runtime_renderer_and_urdf_without_inventing_baseline(tmp_path):
    baseline = make_run(tmp_path / "baseline")
    candidate = make_run(tmp_path / "candidate")
    runtime = {"isaacsim": "5.0.0", "torch": "2.7.0+cu128", "numpy": "1.26.0",
               "scipy": "1.14.1", "Pillow": "11.2.1"}
    renderer = {"light_settle_captures": 16}
    robot_urdf = {"path": "/candidate/vendor/robot.urdf", "bytes": 120, "sha256": "c" * 64}
    update(candidate / "worker_0/eval_manifest.json", runtime_versions=runtime,
           renderer_settings=renderer, robot_urdf=robot_urdf)
    result = MODULE.compare(baseline, candidate)
    before = result["deployment_records"]["baseline"][0]["recorded_runtime"]
    after = result["deployment_records"]["candidate"][0]["recorded_runtime"]
    assert before == {}
    assert after == dict(runtime_versions=runtime, renderer_settings=renderer, robot_urdf=robot_urdf)
    difference = result["deployment_differences"][0]["differences"]["recorded_runtime"]
    assert difference == dict(baseline={}, candidate=after)
