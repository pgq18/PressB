"""Statistical/reporting tests with explicitly synthetic outcome fixtures."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "dppo_experiment_summary", Path(__file__).resolve().parents[1] / "scripts/summarize_dppo_experiment.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def make_run(path, outcomes=(True, False, True, False), *, complete=True, order=None, schedule=True):
    path.mkdir()
    config = dict(method="dppo", mode="eval", seed=7, rollout=dict(
        eval_episodes=len(outcomes), execution_horizon=8, num_envs=2, reset_options={},
        deterministic_eval=True, action_bounds="clip", terminal_observation_mode="full", fused_step_reset=False))
    negotiated = dict(contract=dict(action={"control_hz": 30}, observation={"cameras": ["up", "wrist"]},
        identity=dict(simulation=dict(task_config_sha256="c" * 64, source_revision="d" * 64,
                                      smoothing={"mode": "none"}), inference={"model_id": "test"})),
        action_expert={"denoising_steps": 10}, action_expert_config={})
    if schedule:
        negotiated["seed_schedule"] = "blake2b32(seed:env:episode:stream:step)"
    episodes, resets = [], []
    for completion, index in enumerate(range(len(outcomes)) if order is None else order):
        eid, serial, seed = index % 2, index // 2, index + 100
        uuid = f"{path.name}-{index}"
        episodes.append(dict(transition=completion + 1, env_id=eid, episode_index=serial, reset_seed=seed,
            policy_version=0, episode_id=uuid, success=outcomes[index], terminated=True, truncated=False, info={}))
        resets.append(dict(env_id=eid, episode_index=serial, seed=seed, episode_id=uuid,
                           info={"reset": {"center_xy": [.3 + index * .01, 0.], "color": "red"}}))
    status = dict(state="completed" if complete else "collecting", episodes=len(episodes),
                  successes=sum(outcomes), success_rate=sum(outcomes) / len(outcomes), transitions=len(episodes), updates=0)
    for name, value in (("config.json", config), ("negotiated.json", negotiated), ("status.json", status)):
        (path / name).write_text(json.dumps(value))
    for name, rows in (("episodes.jsonl", episodes), ("resets.jsonl", resets)):
        (path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    return report.load_run(path)


@pytest.mark.parametrize("successes,n,expected", [
    (0, 10, (0., .2775327998628892)), (10, 10, (.7224672001371107, 1.)),
    (5, 10, (.236593090512564, .763406909487436)),
])
def test_wilson_known_values(successes, n, expected):
    assert report.wilson_interval(successes, n) == pytest.approx(expected)
    assert report.wilson_interval(0, 0) is None


@pytest.mark.parametrize("episodes", [1, 3, 10, 100, 1000])
def test_wilson_extreme_outcomes_have_exact_boundary_endpoints(episodes):
    lower, upper = report.wilson_interval(0, episodes)
    assert lower == 0. and 0. < upper < 1.
    lower, upper = report.wilson_interval(episodes, episodes)
    assert 0. < lower < 1. and upper == 1.


@pytest.mark.parametrize("values", [(-1, 4), (5, 4), (True, 4), (1, 0), (1.5, 3)])
def test_wilson_rejects_invalid_counts(values):
    with pytest.raises(ValueError):
        report.wilson_interval(*values)


@pytest.mark.parametrize("improved,worsened,p", [(0, 0, 1.), (8, 0, .0078125), (0, 8, .0078125), (2, 2, 1.), (3, 1, .625)])
def test_exact_mcnemar_known_values(improved, worsened, p):
    assert report.mcnemar_exact(improved, worsened) == pytest.approx(p)


def test_pairing_uses_stable_keys_not_completion_order_or_episode_uuid(tmp_path):
    before = make_run(tmp_path / "before")
    after = make_run(tmp_path / "after", (True, True, False, False), order=[3, 1, 0, 2])
    comparison = report.compare_runs(before, after)
    assert comparison["paired"] and comparison["matched_episodes"] == 4
    assert comparison["improved"] == comparison["worsened"] == 1
    assert comparison["both_success"] == comparison["both_failure"] == 1
    assert comparison["mcnemar_exact_two_sided_p"] == 1
    assert comparison["observed_delta_percentage_points"] == 0


def test_native_elevator_and_pap_termination_keys_are_counted(tmp_path):
    path = tmp_path / 'run'
    make_run(path)
    episodes = [json.loads(line) for line in (path / 'episodes.jsonl').read_text().splitlines()]
    episodes[0]['info'] = {'termination': 'target_pressed'}
    episodes[1]['info'] = {'termination': 'unexpected_collision'}
    episodes[2]['info'] = {'termination_reason': 'success', 'termination': 'legacy'}
    (path / 'episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in episodes))
    assert report.load_run(path)['termination_counts'] == {
        'target_pressed': 1, 'unexpected_collision': 1, 'success': 1, 'unknown': 1}


def test_native_seed_schedule_is_independently_checked(tmp_path):
    before, after = make_run(tmp_path / "before"), make_run(tmp_path / "after")
    for run in (before, after):
        del run["_negotiated"]["seed_schedule"]
        run["_negotiated"]["rollout_seed_schedule"] = dict(algorithm="blake2b", digest_size=4,
            encoding="utf-8", byteorder="little", mask=0x7fffffff,
            preimage="{seed}:{env_id}:{episode_index}:{stream}:{step}", base_seed=7)
        for row in run["_resets"]:
            message = f"7:{row['env_id']}:{row['episode_index']}:policy:0".encode()
            row["policy_seed_at_step0"] = int.from_bytes(hashlib.blake2b(message, digest_size=4).digest(), "little") & 0x7fffffff
    assert report.compare_runs(before, after)["paired"]
    after["_resets"][0]["policy_seed_at_step0"] += 1
    comparison = report.compare_runs(before, after)
    assert not comparison["paired"] and "declared schedule" in comparison["reasons"][0]


@pytest.mark.parametrize("change", ["reset_seed", "layout", "run_seed", "schedule", "missing_resets", "episode_index"])
def test_matching_run_seed_alone_cannot_establish_paired_evidence(tmp_path, change):
    before, after = make_run(tmp_path / "before"), make_run(tmp_path / "after")
    if change == "reset_seed":
        after["_episodes"][0]["reset_seed"] += 1
        after["_resets"][0]["seed"] += 1
    elif change == "layout":
        after["_resets"][0]["info"]["reset"]["center_xy"][0] += .01
    elif change == "run_seed":
        after["_config"]["seed"] += 1
    elif change == "schedule":
        del after["_negotiated"]["seed_schedule"]
    elif change == "missing_resets":
        after["_resets"].clear()
    elif change == "episode_index":
        del after["_episodes"][0]["episode_index"]
    comparison = report.compare_runs(before, after)
    assert comparison["completed_comparison_supported"]
    assert not comparison["paired"] and comparison["reasons"]
    assert "mcnemar_exact_two_sided_p" not in comparison


@pytest.mark.parametrize("change", ["smoothing", "horizon", "sampler", "source", "missing_identity"])
def test_different_eval_protocol_refuses_comparison(tmp_path, change):
    before, after = make_run(tmp_path / "before"), make_run(tmp_path / "after")
    if change == "smoothing":
        after["_negotiated"]["contract"]["identity"]["simulation"]["smoothing"]["mode"] = "ema"
    elif change == "horizon":
        after["_config"]["rollout"]["execution_horizon"] = 16
    elif change == "sampler":
        after["_negotiated"]["action_expert"]["denoising_steps"] = 5
    elif change == "source":
        after["_negotiated"]["contract"]["identity"]["simulation"]["source_revision"] = "changed"
    else:
        del before["_negotiated"]["contract"]["identity"]["simulation"]["source_revision"]
        del after["_negotiated"]["contract"]["identity"]["simulation"]["source_revision"]
    comparison = report.compare_runs(before, after)
    assert not comparison["completed_comparison_supported"] and not comparison["paired"]
    assert comparison["reasons"]


def test_partial_runs_report_observed_counts_without_final_or_paired_claim(tmp_path):
    before, after = make_run(tmp_path / "before"), make_run(tmp_path / "after", complete=False)
    assert after["episodes"] == 4 and after["success_rate"] == .5 and not after["completed"]
    comparison = report.compare_runs(before, after)
    assert not comparison["paired"] and not comparison["completed_comparison_supported"]
    assert after["notes"]


@pytest.mark.parametrize("fault", ["counts", "target", "duplicate", "ongoing_episode", "boolean", "multi_policy"])
def test_completed_logs_are_audited_not_trusted(tmp_path, fault):
    path = tmp_path / "run"
    make_run(path)
    status = json.loads((path / "status.json").read_text())
    episodes = [json.loads(line) for line in (path / "episodes.jsonl").read_text().splitlines()]
    config = json.loads((path / "config.json").read_text())
    if fault == "counts":
        status["successes"] += 1
    elif fault == "target":
        config["rollout"]["eval_episodes"] += 1
    elif fault == "duplicate":
        episodes[1]["episode_id"] = episodes[0]["episode_id"]
    elif fault == "ongoing_episode":
        episodes[0]["terminated"] = False
    elif fault == "boolean":
        episodes[0]["success"] = 1
    else:
        episodes[1]["policy_version"] = 1
    (path / "status.json").write_text(json.dumps(status))
    (path / "config.json").write_text(json.dumps(config))
    (path / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
    with pytest.raises(ValueError):
        report.load_run(path)


def test_report_keeps_sft_loss_separate_and_emits_self_contained_artifacts(tmp_path):
    before = make_run(tmp_path / "before")
    after = make_run(tmp_path / "after", (True, True, True, False), schedule=False)
    sft_path = tmp_path / "sft.jsonl"
    sft_path.write_text(json.dumps(dict(step=1, loss=.5, validation_loss=None)) + "\n" +
                        json.dumps(dict(step=20, loss=.1, validation_loss=.2)) + "\n")
    sft = report.load_sft(sft_path)
    training = deepcopy(before)
    training["_config"]["mode"] = training["mode"] = "train"
    output = tmp_path / "report"
    summary = report.write_report(before, output, after=after, training=training, sft=sft)
    assert set(summary["artifacts"]) == {"evaluation_success.png", "training_curves.png", "sft_loss.png"}
    assert summary["sft"]["validation"] == [{"step": 20, "validation_loss": .2}]
    assert "success_rate" not in summary["sft"]
    assert not summary["comparison"]["paired"]
    text = (output / "README.md").read_text()
    assert "no McNemar result" in text and "not establish" in text
    serialized = json.loads((output / "summary.json").read_text())
    assert serialized == summary and "_episodes" not in serialized["baseline"]
    assert len(serialized["baseline"]["sources"]["episodes.jsonl"]) == 64
    for artifact in summary["artifacts"]:
        assert (output / artifact).read_bytes().startswith(b"\x89PNG")


def test_stdlib_only_report_option_and_strict_json(tmp_path):
    before = make_run(tmp_path / "before")
    report.write_report(before, tmp_path / "no_plot", plots=False)
    assert not list((tmp_path / "no_plot").glob("*.png"))
    bad = tmp_path / "sft.jsonl"
    bad.write_text('{"step": 1, "validation_loss": NaN}\n')
    with pytest.raises(ValueError, match="invalid JSON"):
        report.load_sft(bad)


def test_plot_report_handles_zero_and_full_success_without_negative_errorbars(tmp_path):
    before = make_run(tmp_path / "before", outcomes=(True,) * 100)
    after = make_run(tmp_path / "after", outcomes=(False,) * 100)
    output = tmp_path / "report"
    result = report.write_report(before, output, after=after)
    assert result["baseline"]["wilson_95"][1] == 1.
    assert result["after"]["wilson_95"][0] == 0.
    assert result["comparison"]["paired"]
    assert result["comparison"]["worsened"] == 100
    assert result["artifacts"] == ["evaluation_success.png"]
    assert (output / "evaluation_success.png").read_bytes().startswith(b"\x89PNG")


def test_sft_does_not_count_carried_validation_values_as_new_evaluations(tmp_path):
    (tmp_path / "resolved_config.json").write_text(json.dumps(dict(eval_every=500, max_steps=750)))
    rows = [dict(step=step, validation_loss=loss) for step, loss in ((500, .2), (525, .2), (550, .2), (750, .1))]
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = report.load_sft(tmp_path)
    assert result["validation"] == [rows[0], rows[-1]]
    assert result["validation_points"] == "scheduled evaluations"
