import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("collection_suite", ROOT / "scripts/run_table_buttons_collection_suite.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def fixture_source(tmp_path, count=2):
    source = tmp_path / "old"
    for profile in suite.PROFILES:
        shard = source / "raw_shards" / profile
        suite.write(shard / "plan.json", dict(profiles=[profile], targets=list(suite.COLORS), seed_start=0,
            max_retries=50, episodes_per_group=count, config={"fps": 30, "physics_dt": 1/120}))
        for index in range(count):
            suite.write(shard / "layouts" / f"{profile}_seed{index}.json", {"seed": index})
        for color in suite.COLORS:
            group = f"{profile}_{color}"
            for index in range(count):
                path = shard / group / f"episode_{index:06d}"
                path.mkdir(parents=True)
                for filename in suite.EPISODE_FILES:
                    (path / filename).write_text("fixture")
                suite.write(path / "metadata.json", dict(success=True, group=group, episode_index=index,
                    seed=index, positions_xy_m={"red": [index, 0], "yellow": [index, 1], "blue": [index, 2]}))
    return source


def args_for(tmp_path, source):
    return SimpleNamespace(source=source, output=tmp_path / "new", dataset_root=tmp_path / "datasets",
        reuse_per_group=2, episodes_per_group=4, max_new_episodes=50, gpu=1,
        max_infrastructure_restarts=3, poll_seconds=.01)


def test_prepare_real_copies_immutable_prefix_and_plan_only_quota_changes(tmp_path, monkeypatch):
    source = fixture_source(tmp_path)
    args = args_for(tmp_path, source)
    monkeypatch.setattr(suite, "frozen_paths", lambda: [ROOT / "scripts/collect_table_buttons.py"])
    hashes = {str(path): suite.digest(path) for path in source.rglob("*") if path.is_file()}
    result = suite.prepare(args)
    assert result["reused_episodes"] == 18 and result["new_episodes"] == 18
    assert result["requested_episodes"] == 36
    suite.prepare(args)  # restartable preparation is idempotent
    assert hashes == {str(path): suite.digest(path) for path in source.rglob("*") if path.is_file()}
    for profile in suite.PROFILES:
        old = source / "raw_shards" / profile
        new = args.output / "raw_shards" / profile
        assert suite.read(new / "plan.json") == dict(suite.read(old / "plan.json"), episodes_per_group=4)
        for color in suite.COLORS:
            group = f"{profile}_{color}"
            assert (args.output / "raw" / group).resolve() == new / group
            for index in range(2):
                before, after = [base / group / f"episode_{index:06d}" / "data.npz" for base in (old, new)]
                assert before.read_bytes() == after.read_bytes()
                assert before.stat().st_ino != after.stat().st_ino and not after.is_symlink()


def test_drift_in_frozen_config_is_rejected(tmp_path, monkeypatch):
    args = args_for(tmp_path, fixture_source(tmp_path))
    monkeypatch.setattr(suite, "frozen_paths", lambda: [])
    suite.prepare(args)
    suite.write(args.output / "raw_shards/near/frozen_config.json", {"changed": True})
    with pytest.raises(RuntimeError, match="config changed"):
        suite.verify_code(args.output)


@pytest.mark.parametrize("code,state,reason,count,want", [
    (75, "restart_required", "release_cpu_stage_caches", 2, "rotation"),
    (75, "restart_required", "requested", 2, "stop_requested"),
    (75, "failed", "release_cpu_stage_caches", 2, "infrastructure_failure"),
    (75, "restart_required", "unknown", 2, "infrastructure_failure"),
    (0, "complete", None, 12, "complete"),
    (0, "complete", None, 11, "infrastructure_failure"),
    (1, "failed", None, 2, "infrastructure_failure"),
])
def test_rotation_and_completion_require_matching_exit_and_status(code, state, reason, count, want):
    assert suite.classify_collector_exit(code, {"state": state, "restart_reason": reason}, count, 12) == want
    assert suite.classify_collector_exit(1, {"error": "Failed to obtain near_red episode 2 within retry budget"}, 2, 12) == "retry_budget_exhausted"


def test_index_only_counts_committed_and_reports_independent_color_retry(tmp_path, monkeypatch):
    args = args_for(tmp_path, fixture_source(tmp_path))
    monkeypatch.setattr(suite, "frozen_paths", lambda: [])
    suite.prepare(args)
    (args.output / "raw_shards/near/pending/near_red/episode_000002_seed2").mkdir(parents=True)
    groups = suite.EpisodeIndex().scan(args.output, 4)
    assert sum(map(len, groups.values())) == 18
    assert suite.pairing_report(groups)["mismatch_count"] == 0
    groups["near_red"][1] = dict(groups["near_red"][1], seed=1001001)
    report = suite.pairing_report(groups)
    assert report["mismatch_count"] == 1 and report["mismatches"][0]["seeds"]["red"] == 1001001
    (args.output / "raw_shards/near/near_red/episode_000004").mkdir()
    with pytest.raises(ValueError, match="Nonconsecutive"):
        suite.EpisodeIndex().scan(args.output, 4)


def test_export_complete_uses_actual_manifest_and_audit_contract(tmp_path):
    destination, raw = tmp_path / "export", tmp_path / "raw"
    suite.write(destination / "meta/export_manifest.json", dict(group="near_red", raw_root=str(raw), episodes=[{}, {}]))
    suite.write(destination / "meta/audit.json", dict(success=True, episodes=2))
    suite.write(destination / "meta/info.json", dict(total_episodes=2))
    assert suite.exported_complete(destination, "near_red", 2, raw)
    assert not suite.exported_complete(destination, "near_red", 3, raw)
    assert not suite.exported_complete(destination, "near_blue", 2, raw)
    suite.write(destination / "meta/audit.json", dict(success=False, episodes=2))
    assert not suite.exported_complete(destination, "near_red", 2, raw)


def test_command_preserves_seed_rotation_config_and_physical_gpu_namespace(tmp_path, monkeypatch):
    args = args_for(tmp_path, tmp_path / "old")
    command = suite.collection_command(args, "near")
    assert command[command.index("--episodes-per-group")+1] == "4"
    assert command[command.index("--seed-start")+1] == "0"
    assert command[command.index("--max-retries")+1] == "50"
    assert command[command.index("--max-new-episodes")+1] == "50"
    assert command[command.index("--config")+1] == str(args.output / "raw_shards/near/frozen_config.json")
    assert command[-1] == "--resume"
    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.setenv("XAUTHORITY", "/tmp/example.Xauthority")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONPATH", "/tmp/foreign")
    env = suite.environment()
    assert env["DISPLAY"] == ":1" and env["XAUTHORITY"] == "/tmp/example.Xauthority"
    assert "CUDA_VISIBLE_DEVICES" not in env and "PYTHONPATH" not in env
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "1"


def test_lock_rejects_duplicate_supervisor(tmp_path):
    with suite.exclusive_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another supervisor"):
            with suite.exclusive_lock(tmp_path):
                pass


def fresh_args(tmp_path):
    config_path = tmp_path / "motion.json"
    suite.write(config_path, {"fps": 30, "physics_dt": 1/120,
                             "motion_templates": {"mode": "target_relative_v2"}})
    args = args_for(tmp_path, None)
    args.config, args.reuse_per_group = config_path, 0
    return args


def test_fresh_suite_has_nine_empty_groups_frozen_config_and_resume_contract(tmp_path, monkeypatch):
    args = fresh_args(tmp_path)
    monkeypatch.setattr(suite, "frozen_paths", lambda: [])
    contract = suite.prepare(args)
    assert contract["source"] is None
    assert contract["requested_episodes"] == contract["new_episodes"] == 36
    assert contract["reused_episodes"] == 0
    assert suite.prepare(args) == contract
    assert sum(map(len, suite.EpisodeIndex().scan(args.output, 4).values())) == 0
    for profile in suite.PROFILES:
        shard = args.output / "raw_shards" / profile
        assert suite.read(shard / "plan.json") == {
            "schema_version": "pressb.table_buttons.collection_plan.v1",
            "config": suite.read(args.config), "profiles": [profile], "targets": list(suite.COLORS),
            "episodes_per_group": 4, "seed_start": 0, "max_retries": 50, "fps": 30, "physics_hz": 120,
            "instructions": {color: f"Press the {color} button." for color in suite.COLORS},
            "seed_rule": "first attempt seed_start+episode_index; retry 1000000+1000*(seed_start+episode_index)+retry",
            "success_only": True, "demonstration_end": "first control boundary after physical success",
        }
        assert suite.read(shard / "frozen_config.json") == suite.read(args.config)
        assert list((shard / "layouts").iterdir()) == []
        for color in suite.COLORS:
            assert (args.output / "raw" / f"{profile}_{color}").resolve() == shard / f"{profile}_{color}"
    manifest = suite.read(args.output / "import_manifest.json")
    assert manifest["copied_episodes"] == manifest["copied_bytes"] == 0 and manifest["files"] == {}
    suite.verify_code(args.output)
    suite.write(args.output / "raw_shards/near/frozen_config.json", {"changed": True})
    with pytest.raises(RuntimeError, match="config changed"):
        suite.verify_code(args.output)


@pytest.mark.parametrize("invalid", ["source", "reuse", "quota", "fps", "physics_dt"])
def test_fresh_collection_rejects_mixed_prefix_or_invalid_contract(tmp_path, monkeypatch, invalid):
    args = fresh_args(tmp_path)
    monkeypatch.setattr(suite, "frozen_paths", lambda: [])
    if invalid == "source":
        args.source = tmp_path / "old"
    elif invalid == "reuse":
        args.reuse_per_group = 1
    elif invalid == "quota":
        args.episodes_per_group = 0
    else:
        config = suite.read(args.config)
        config[invalid] = 60 if invalid == "fps" else 1/60
        suite.write(args.config, config)
    with pytest.raises(ValueError, match="Fresh collection requires"):
        suite.prepare(args)
    assert not (args.output / "dataset_plan.json").exists()


def test_fresh_resume_rejects_changed_input_config(tmp_path, monkeypatch):
    args = fresh_args(tmp_path)
    monkeypatch.setattr(suite, "frozen_paths", lambda: [])
    suite.prepare(args)
    config = suite.read(args.config)
    config["motion_templates"]["mode"] = "shared"
    suite.write(args.config, config)
    with pytest.raises(ValueError, match="Existing preparation differs"):
        suite.prepare(args)


def test_per_profile_gpu_mapping_overrides_default(tmp_path):
    args = args_for(tmp_path, tmp_path / "old")
    args.profile_gpus = [0, 1, 1]
    for profile, gpu in zip(suite.PROFILES, args.profile_gpus):
        command = suite.collection_command(args, profile)
        assert command[command.index("--gpu") + 1] == str(gpu)
