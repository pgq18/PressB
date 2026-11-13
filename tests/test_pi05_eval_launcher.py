"""Pi05 config preparation and queue forwarding without models, CUDA or Isaac."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    SPEC = importlib.util.spec_from_file_location("pi05_eval_launcher", ROOT / "scripts/run_pi05_manipulation_eval.py")
    launcher = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(launcher)
finally:
    sys.path.pop(0)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    checkpoint = tmp_path / "pretrained_model"
    launcher.write_json(checkpoint / "config.json", dict(type="pi05", train_expert_only=True, chunk_size=50))
    launcher.write_json(checkpoint / "pressb_pi05.json", dict(
        format="pressb.pi05.v1", checkpoint_step=20,
        observation=dict(cameras=["up", "wrist"]),
        action=dict(representation="joint_position", control_hz=30, units=["rad"] * 6 + ["m"])))
    (checkpoint / "model.safetensors").write_bytes(b"fake model; preparation must not load it")
    task = deepcopy(launcher.read_json(ROOT / "configs/experiments/cube_stack_pi05_eval.json")["task"]["config"])
    task.update(scene_config="configs/calibration/playground_20261109_global_refined.json",
                num_envs=16, horizon=32, max_episode_seconds=150, terminate_on_success=False)
    task["scene_overrides"]["rendering"]["warmup_frames"] = 6
    dataset_root = tmp_path / "datasets"
    for suffix in ("500_20261110", "100_20261113"):
        launcher.write_json(dataset_root / f"pressb_cube_stack_{suffix}/meta/export_manifest.json",
                            dict(task_id="cube_stack", fps=30, raw_manifest=dict(config=task)))
    monkeypatch.setattr(launcher, "gpu_snapshot", lambda: pytest.fail("Config preparation queried GPU state"))
    monkeypatch.setattr(launcher.Pi05Evaluation, "launch", lambda *a: pytest.fail("Config preparation launched a process"))
    monkeypatch.setattr(launcher.signal, "signal", lambda *a: None)
    return SimpleNamespace(root=tmp_path, checkpoint=checkpoint, dataset_root=dataset_root, task=task)


def arguments(inputs, *extra):
    return ["evaluate", "--task", "cube_stack", "--checkpoint", str(inputs.checkpoint),
            "--dataset-root", str(inputs.dataset_root), "--output", str(inputs.root / "output"), *extra]


@pytest.mark.parametrize("profile,batch", [("legacy", 4), ("fast", 8)])
def test_profiles_prepare_valid_configs_without_gpu_and_preserve_eval_semantics(inputs, profile, batch):
    assert launcher.main(arguments(inputs, "--performance-profile", profile, "--prepare-config-only")) == 0
    config = launcher.read_json(inputs.root / "output/experiment.json")
    task, run = config["task"]["config"], config["run"]
    assert task["scene_config"] == str(ROOT / inputs.task["scene_config"])
    for key in ("render_resolution", "image_output_resolution", "scene_overrides", "randomization",
                "observation_render_frames", "reset_render_frames", "settling_ticks", "smoothing"):
        assert task[key] == inputs.task[key]
    assert task["horizon"] == 50 and task["num_envs"] == 8
    assert task["max_episode_seconds"] == 45 and task["terminate_on_success"] is True
    assert run["seed"] == 2026111101 and run["endpoints"]["timeout"] == 300
    assert run["rollout"]["execution_horizon"] == 7 and run["rollout"]["eval_episodes"] == 100
    assert run["rollout"]["max_transitions"] == 135000
    service = config["nodes"]["inference"]["service"]
    assert service["batch_size"] == batch
    assert task.get("color_response_workers", 0) == (4 if profile == "fast" else 0)
    assert service.get("collect_timing", False) is (profile == "fast")
    assert run["rollout"]["terminal_observation_mode"] == ("state_only_if_no_bootstrap" if profile == "fast" else "full")
    assert run["rollout"]["fused_step_reset"] is (profile == "fast")
    status = launcher.read_json(inputs.root / "output/status.json")
    assert status["status"] == "validated" and status["gpu_processes_started"] is False
    assert status["children_cleaned_up"] is True
    assert status["evaluation_request"]["dataset_manifest"].endswith("pressb_cube_stack_500_20261110/meta/export_manifest.json")


@pytest.mark.parametrize("profile", ["legacy", "fast"])
def test_explicit_batch_size_overrides_profile(inputs, profile):
    parsed = launcher.parse_args(arguments(inputs, "--performance-profile", profile, "--batch-size", "3"))
    assert parsed.batch_size == 3


@pytest.mark.parametrize("selection", ["path", "manifest", "name", "suffix"])
def test_selects_new_100_episode_dataset_without_rewriting_its_manifest(inputs, selection):
    dataset = inputs.dataset_root / "pressb_cube_stack_100_20261113"
    manifest = dataset / "meta/export_manifest.json"
    original = manifest.read_bytes()
    flags = {"path": ["--dataset", str(dataset)], "manifest": ["--dataset", str(manifest)],
             "name": ["--dataset-name", dataset.name], "suffix": ["--dataset-suffix", "100_20261113"]}[selection]
    assert launcher.main(arguments(inputs, *flags, "--episodes", "3", "--validate-only")) == 0
    status = launcher.read_json(inputs.root / "output/status.json")
    assert status["evaluation_request"]["dataset_manifest"] == str(manifest)
    assert status["expected_episodes"] == 3 and status["mode"] == "diagnostic"
    assert manifest.read_bytes() == original


def test_task_override_is_explicit_and_deep_merges_onto_selected_manifest(inputs):
    override = inputs.root / "task.json"
    launcher.write_json(override, dict(scene_config="configs/calibration/playground_20261113_appearance.json",
                                     scene_overrides=dict(rendering=dict(warmup_frames=9))))
    assert launcher.main(arguments(inputs, "--task-config", str(override), "--validate-only")) == 0
    config = launcher.read_json(inputs.root / "output/experiment.json")["task"]["config"]
    assert config["scene_config"] == str(ROOT / "configs/calibration/playground_20261113_appearance.json")
    assert config["scene_overrides"]["rendering"]["warmup_frames"] == 9
    assert config["randomization"] == inputs.task["randomization"]
    assert config["horizon"] == 50 and config["max_episode_seconds"] == 45


@pytest.mark.parametrize("field,value", [("task_id", "tabletop_cleanup"), ("fps", 15)])
def test_dataset_must_match_task_and_checkpoint_rate(inputs, field, value):
    manifest = inputs.dataset_root / "pressb_cube_stack_500_20261110/meta/export_manifest.json"
    data = launcher.read_json(manifest)
    data[field] = value
    launcher.write_json(manifest, data)
    with pytest.raises(ValueError, match="Dataset manifest differs"):
        launcher.main(arguments(inputs, "--validate-only"))


@pytest.mark.parametrize("change", [dict(train_expert_only=False), dict(type="other"), dict(chunk_size=0)])
def test_checkpoint_metadata_is_verified_before_preparing(inputs, change):
    path = inputs.checkpoint / "config.json"
    launcher.write_json(path, {**launcher.read_json(path), **change})
    with pytest.raises(ValueError, match="expert-only"):
        launcher.main(arguments(inputs, "--validate-only"))
    assert not (inputs.root / "output/experiment.json").exists()


def test_queue_forwards_dataset_episode_runtime_and_performance_options(tmp_path):
    args = launcher.parse_args(["queue", "--remote-root", "/remote", "--queue-root", str(tmp_path / "queue"),
        "--archive-root", str(tmp_path / "archives"), "--dataset-root", str(tmp_path / "datasets"),
        "--dataset-suffix", "100_20261113", "--performance-profile", "fast", "--batch-size", "5",
        "--episodes", "23", "--num-envs", "10", "--simulation-workers", "2", "--sim-gpu", "2",
        "--inference-gpu", "3", "--simulation-port", "19001", "--inference-port", "19000",
        "--poll-seconds", ".5", "--startup-timeout", "50", "--min-free-mib", "0",
        "--sim-python", "/sim/python", "--model-python", "/model/python"])
    command = launcher.queue_command(args, "tabletop_cleanup", tmp_path / "archive.json", tmp_path / "out")
    child = launcher.parse_args(command[3:])
    assert launcher.dataset_manifest(child) == tmp_path / "datasets/pressb_tabletop_cleanup_100_20261113/meta/export_manifest.json"
    for name in ("episodes", "num_envs", "batch_size", "performance_profile", "simulation_workers",
                 "sim_gpu", "inference_gpu", "sim_python", "model_python", "inference_port",
                 "simulation_port", "min_free_mib", "startup_timeout", "poll_seconds"):
        assert getattr(child, name) == getattr(args, name)


def test_result_reuse_requires_same_episode_count_dataset_and_profile(inputs):
    args = launcher.parse_args(arguments(inputs, "--episodes", "16"))
    folder = inputs.root / "queue"
    launcher.write_json(folder / "eval_001/status.json", dict(status="completed", children_cleaned_up=True))
    launcher.write_json(folder / "eval_001/result.json", dict(task="cube_stack", mode="diagnostic", episodes=16,
        evaluation_request=launcher.evaluation_request(args)))
    assert launcher.completed_result("cube_stack", folder, args) is not None
    for extra in (("--episodes", "17"), ("--dataset-suffix", "100_20261113"), ("--performance-profile", "fast")):
        other = launcher.parse_args(arguments(inputs, "--episodes", "16", *extra))
        assert launcher.completed_result("cube_stack", folder, other) is None


def test_two_workers_delegate_validation_with_verified_identity_and_per_worker_env_budget(inputs, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "run_sharded_evaluation", SimpleNamespace(main=lambda argv: calls.append(argv) or 0))
    assert launcher.main(arguments(inputs, "--simulation-workers", "2", "--performance-profile", "fast",
                                    "--prepare-config-only")) == 0
    command, = calls
    values = {command[i]: command[i + 1] for i in range(0, len(command) - 1, 2)}
    assert values["--workers"] == "2" and values["--num-envs"] == "8"
    assert values["--episodes"] == "100" and values["--batch-size"] == "8"
    assert values["--expected-task-id"] == "cube_stack"
    assert values["--expected-weights-sha256"] == launcher.sha256(inputs.checkpoint / "model.safetensors")
    assert command[-1] == "--validate-only"
    status = launcher.read_json(inputs.root / "output/status.json")
    assert status["total_envs"] == 16 and status["num_envs_per_worker"] == 8


def test_two_workers_prepare_real_sharded_configs_without_gpu(inputs, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    assert launcher.main(arguments(inputs, "--simulation-workers", "2", "--performance-profile", "fast",
                                    "--prepare-config-only")) == 0
    for index in range(2):
        config = launcher.read_json(inputs.root / f"output/sharded/workers/{index:03d}/experiment.json")
        assert config["task"]["config"]["num_envs"] == 8
        assert config["run"]["rollout"]["eval_episodes"] == 50
        assert config["run"]["seed"] == launcher.TASK_SEEDS["cube_stack"] + index
        assert config["run"]["rollout"]["execution_horizon"] == 7
        assert config["task"]["config"]["max_episode_seconds"] == 45
        assert config["nodes"]["inference"]["server"]["port"] == 18914
        assert config["nodes"]["simulation"]["server"]["port"] == 18915 + index


@pytest.mark.parametrize("flags", [
    ["--dataset", "/data", "--dataset-suffix", "100_20261113"],
    ["--dataset-name", "../other"], ["--dataset-suffix", "../other"],
    ["--simulation-workers", "2", "--performance-profile", "fast", "--episodes", "15"],
    ["--simulation-workers", "2", "--episodes", "1"],
    ["--simulation-workers", "2", "--inference-port", "18916"],
    ["--batch-size", "0"], ["--episodes", "0"], ["--startup-timeout", "nan"],
])
def test_invalid_or_ambiguous_arguments_fail_before_launch(inputs, flags):
    with pytest.raises(SystemExit):
        launcher.parse_args(arguments(inputs, *flags))


def test_queue_direct_dataset_requires_task_placeholder(tmp_path):
    with pytest.raises(SystemExit):
        launcher.parse_args(["queue", "--remote-root", "/remote", "--queue-root", str(tmp_path / "q"),
                             "--archive-root", str(tmp_path / "a"), "--dataset", "/one-task-only"])


@pytest.mark.parametrize("failure", [None, "context_cleanup_error", "context_cleanup_failed",
                                     "simulation_close_failed", "observer_close_failed"])
def test_single_worker_checks_cleanup_after_successful_process_exit(inputs, monkeypatch, failure):
    args = launcher.parse_args(arguments(inputs, "--episodes", "1"))
    args.smoke_run = None
    runner = launcher.Pi05Evaluation(args)
    checkpoint = launcher.inspect_checkpoint(inputs.checkpoint)
    launcher.write_json(args.output / "inference.health.json", dict(weights_sha256=checkpoint["model_sha256"]))
    launcher.write_json(args.output / "simulation.health.json", dict(task_id="cube_stack"))
    status = dict(state="completed", episodes=1, successes=1)
    if failure == "context_cleanup_error":
        status["context_cleanup_error"] = "could not release context"
    launcher.write_json(args.output / "run/status.json", status)
    row = dict(env_id=0, episode_id="episode-0", success=True,
               info=dict(physical_outcome=dict(success=True), terminal_reason="success"))
    (args.output / "run/episodes.jsonl").write_text(json.dumps(row) + "\n")
    if failure and failure != "context_cleanup_error":
        (args.output / "run/events.jsonl").write_text(json.dumps(dict(event=failure, error="cleanup failed")) + "\n")

    def launch(name, *_):
        runner.children[name] = SimpleNamespace(poll=lambda: 0 if name == "eval" else None, returncode=0)

    monkeypatch.setattr(runner, "launch", launch)
    monkeypatch.setattr(runner, "await_health", lambda *_: None)
    monkeypatch.setattr(launcher, "port_available", lambda *_: None)
    if failure:
        with pytest.raises(RuntimeError, match="Evaluation cleanup failed"):
            runner.run_single(args.output / "experiment.json", {}, checkpoint)
    else:
        result = runner.run_single(args.output / "experiment.json", {}, checkpoint)
        assert result["episodes"] == 1 and result["success_rate"] == 1.0
