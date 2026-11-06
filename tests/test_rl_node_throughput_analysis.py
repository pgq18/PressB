"""Prevent misleading throughput claims from warmup, lost UTD or partial logs."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analyze_rl_node_throughput",
                                            ROOT / "scripts/analyze_rl_node_throughput.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def metric(transitions, elapsed, updates, *, backlog=0., episodes=0, successes=0, resets=1):
    return dict(run_id="training-run", schema_fingerprint="model+task", state="running",
                transitions=transitions, elapsed_seconds=elapsed, updates=updates,
                update_backlog=backlog, episodes=episodes, successes=successes,
                dropped_update_credit=0., learner_seconds=updates*.02,
                last_update_metrics={"critic_loss": .2},
                rpc_seconds={"simulation/step": elapsed*.3, "inference/encode": elapsed*.65,
                             "simulation/reset": resets*.1},
                rpc_calls={"simulation/step": transitions//64, "inference/encode": transitions//64,
                           "simulation/reset": resets})


def fixture():
    rows = [metric(64, 30, 0), metric(2048, 200, 64),
            metric(4096, 240, 2112), metric(8192, 600, 6208, episodes=2, successes=1, resets=3)]
    episodes = [dict(transition=5000, env_id=0, episode_id="success", success=True,
                     terminated=True, truncated=False, info={"termination_reason": "success"}),
                dict(transition=8000, env_id=1, episode_id="timeout", success=False,
                     terminated=False, truncated=True, info={"termination_reason": "timeout"})]
    config = dict(mode="train", update={"learning_starts": 2000, "utd": 1.}, learner={"batch_size": 256})
    return rows, episodes, config


def test_measures_transition_difference_over_total_time_and_counts_episodes_separately():
    rows, episodes, config = fixture()
    report = analysis.analyze(rows, episodes, config)
    assert report["window"]["start_transitions"] == 2048
    assert report["window"]["transitions"] == 6144
    assert report["window"]["wall_seconds"] == 400
    assert report["window"]["transitions_per_second"] == 15.36
    assert report["episodes"]["completed"] == 2
    assert report["learning"]["completed_utd"] == 1.
    assert report["qualification"]["passed"]
    assert report["projection"]["400000"]["hours"] == pytest.approx(400000/15.36/3600)
    assert report["learning"]["learner_seconds_overlap_rpc"]
    # Learner wall time overlaps RPC durations; there is no misleading exclusive sum.
    assert report["window"]["wall_seconds"] != sum(row["seconds"] for row in report["rpc"].values()) \
        + report["learning"]["learner_seconds"]


def test_short_fast_window_does_not_qualify_and_does_not_include_pre_warmup_rate():
    rows, episodes, config = fixture()
    report = analysis.analyze(rows, episodes, config, last_seconds=365.)
    assert report["window"]["start_transitions"] == 4096
    assert report["window"]["transitions_per_second"] == pytest.approx(4096/360)
    assert not report["qualification"]["checks"]["sufficient_transitions"]
    assert not report["qualification"]["passed"]
    with pytest.raises(ValueError, match="two records after warmup"):
        analysis.analyze(rows, episodes, config, after_transition=8192)


@pytest.mark.parametrize("failure", ["dropped_credit", "backlog", "accounting", "nonfinite_loss", "episodes"])
def test_no_false_acceptance_when_update_work_or_evidence_is_missing(failure):
    rows, episodes, config = fixture()
    if failure == "dropped_credit":
        rows[-1]["dropped_update_credit"] = 1024
        rows[-1]["updates"] -= 1024
    elif failure == "backlog":
        rows[-1]["update_backlog"] = 1024
        rows[-1]["updates"] -= 1024
    elif failure == "accounting":
        rows[-1]["updates"] -= 5
    elif failure == "nonfinite_loss":
        rows[2]["last_update_metrics"]["critic_loss"] = float("nan")
    elif failure == "episodes":
        episodes.pop()
    report = analysis.analyze(rows, episodes, config)
    assert not report["qualification"]["passed"]


@pytest.mark.parametrize("failure", ["run_id", "schema_fingerprint", "elapsed_seconds", "rpc_seconds"])
def test_rejects_merged_sessions_and_counter_resets(failure):
    rows, episodes, config = fixture()
    if failure in {"run_id", "schema_fingerprint"}:
        rows[-1][failure] = "other"
    elif failure == "elapsed_seconds":
        rows[-1][failure] = 1.
    else:
        rows[-1][failure]["inference/encode"] = 0.
    with pytest.raises(ValueError, match="Mixed run|regressed"):
        analysis.analyze(rows, episodes, config)


def test_rejects_duplicate_episode_rows():
    rows, episodes, config = fixture()
    episodes.append(deepcopy(episodes[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        analysis.analyze(rows, episodes, config)


def test_live_tail_does_not_hide_interior_corruption(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b'{"transitions": 64}\n{"transitions":')
    rows, source = analysis.read_jsonl(path)
    assert rows == [{"transitions": 64}]
    assert source["ignored_incomplete_final_line"]
    assert len(source["sha256"]) == 64
    path.write_bytes(b'{"transitions": 64}\ncorrupted\n{"transitions":')
    with pytest.raises(ValueError, match="line 2"):
        analysis.read_jsonl(path)


def test_final_checkpoint_time_is_included(tmp_path, monkeypatch):
    rows, episodes, config = fixture()
    for name, value in (("metrics", rows), ("episodes", episodes)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in value))
    (tmp_path / "config.json").write_text(json.dumps(config))
    final = {**rows[-1], "elapsed_seconds": 650, "state": "completed"}
    (tmp_path / "status.json").write_text(json.dumps(final))
    output = tmp_path / "analysis.json"
    monkeypatch.setattr("sys.argv", ["analyze", "--run", str(tmp_path), "--output", str(output)])
    analysis.main()
    report = json.loads(output.read_text())
    assert report["window"]["wall_seconds"] == 450
    assert report["window"]["includes_final_status_time"]
    assert report["window"]["transitions_per_second"] == pytest.approx(6144/450)


def test_resources_explicitly_do_not_assume_clock_alignment():
    rows = [{"elapsed_seconds": 1, "host_available_bytes": 10, "process_tree_rss_bytes": 20,
             "gpu": {"devices": [{"uuid": "gpu0", "index": 0, "used_mib": 10, "free_mib": 50}]}},
            {"elapsed_seconds": 2, "host_available_bytes": 5, "process_tree_rss_bytes": 30,
             "gpu": {"devices": [{"uuid": "gpu0", "index": 0, "used_mib": 20, "free_mib": 40}]}}]
    report = analysis.resource_summary(rows)
    assert "not_time_aligned" in report["scope"]
    assert report["minimum_host_available_bytes"] == 5
    assert report["devices"]["gpu0"]["peak_used_mib"] == 20


def fused_fixture(*, count_resets=True, reset_envs=2):
    rows, episodes, config = fixture()
    for row in rows:
        calls = row["rpc_calls"].pop("simulation/step")
        seconds = row["rpc_seconds"].pop("simulation/step")
        row["rpc_calls"]["simulation/step_and_reset"] = calls
        row["rpc_seconds"]["simulation/step_and_reset"] = seconds
        # Only the initial all-env reset uses a separate RPC. It precedes the
        # measurement window and cannot establish any subsequent reset work.
        row["rpc_calls"]["simulation/reset"] = 1
        row["rpc_seconds"]["simulation/reset"] = .1
        if count_resets:
            row.update(reset_envs=64, reset_calls=1)
    if count_resets:
        rows[-1]["reset_envs"] += reset_envs
        rows[-1]["reset_calls"] += int(reset_envs > 0)
    return rows, episodes, config


def test_fused_reset_uses_validated_reset_items_not_rpc_name():
    report = analysis.analyze(*fused_fixture())
    assert report["qualification"]["passed"]
    assert report["resets"]["environment_resets"] == 2
    assert report["resets"]["actual_reset_batches"] == 1
    assert report["resets"]["explicit_reset_rpc_calls"] == 0
    assert report["resets"]["fused_step_rpc_calls"] == 96
    assert report["resets"]["evidence_source"] == "validated_reset_items_counters"


@pytest.mark.parametrize("count_resets", [False, True])
def test_fused_calls_without_proven_reset_items_cannot_pass(count_resets):
    report = analysis.analyze(*fused_fixture(count_resets=count_resets, reset_envs=0))
    assert report["resets"]["fused_step_rpc_calls"] == 96
    assert not report["qualification"]["checks"]["indexed_resets_observed"]
    assert not report["qualification"]["passed"]


def test_legacy_reset_evidence_is_a_window_delta_not_initial_reset_count():
    rows, episodes, config = fixture()
    report = analysis.analyze(rows, episodes, config)
    assert report["resets"]["evidence_source"] == "legacy_explicit_reset_rpc"
    assert report["resets"]["environment_resets"] is None
    assert report["resets"]["actual_reset_batches"] == 2
    for row in rows:
        row["rpc_calls"]["simulation/reset"] = 64
        row["rpc_seconds"]["simulation/reset"] = 100
    report = analysis.analyze(rows, episodes, config)
    assert not report["qualification"]["checks"]["indexed_resets_observed"]


@pytest.mark.parametrize("failure", ["missing", "regressed", "boolean", "fractional", "schema_changed",
                                     "calls_without_items", "items_without_calls"])
def test_bad_reset_counters_fail_closed(failure):
    rows, episodes, config = fused_fixture()
    if failure == "missing":
        rows[-1].pop("reset_calls")
    elif failure == "regressed":
        rows[-1]["reset_envs"] = 63
    elif failure == "boolean":
        rows[-1]["reset_envs"] = True
    elif failure == "fractional":
        rows[-1]["reset_envs"] = 65.5
    elif failure == "schema_changed":
        rows[-1].pop("reset_envs")
        rows[-1].pop("reset_calls")
    elif failure == "calls_without_items":
        rows[-1]["reset_envs"] = 64
    else:
        rows[-1]["reset_calls"] = 1
    with pytest.raises(ValueError, match="reset counter|Reset counter"):
        analysis.analyze(rows, episodes, config)
