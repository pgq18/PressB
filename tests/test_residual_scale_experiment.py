"""Half-scale experiment contracts, with CPU-only rollout and SAC checks."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pressb.online_rl.learner import LearnerConfig, SACLearner
from pressb.online_rl.protocol import pose9_to_pose8
from pressb.online_rl.residual_on_noise import ResidualOnNoiseRunner, file_sha256
from test_fast_online_rl_runner import BatchInference, config
from test_online_rl_runner import TOKEN, serve
from test_residual_on_noise import (
    GammaIndexedSimulation, noise_checkpoint, prepared_runner,
)


ROOT = Path(__file__).resolve().parents[1]
HALF_SCALE = [.015, .015, .015, .05, .05, .05, .05, .05, .05]
CONFIG = ROOT / "configs/online_rl_residual_on_noise_gamma0995_scale050_400k.json"
spec = importlib.util.spec_from_file_location(
    "residual_scale_experiment_supervisor", ROOT / "scripts/run_residual_on_noise_experiment.py")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def load_config(tmp_path, value):
    path = tmp_path / "selected.json"
    path.write_text(json.dumps(value))
    experiment = supervisor.Experiment(tmp_path / "run", tmp_path / "noise.pt", config=path)
    return experiment.load_config()


def half_scale_config():
    return json.loads(CONFIG.read_text())


def test_half_scale_config_changes_only_scale_from_previous_gamma0995_run(tmp_path):
    selected = half_scale_config()
    assert selected["learner"]["residual_scale"] == HALF_SCALE
    assert selected["single_gamma"] == .995
    assert selected["max_transitions"] == 400_000
    expected = json.loads((ROOT / "configs/online_rl_residual_on_noise_gamma0995_400k.json").read_text())
    assert np.array_equal(np.asarray(selected["learner"]["residual_scale"]) * 2,
                          expected["learner"]["residual_scale"])
    expected["learner"]["residual_scale"] = HALF_SCALE
    assert selected == expected
    assert load_config(tmp_path, selected) == selected


@pytest.mark.parametrize("scale", [
    [], HALF_SCALE[:8], HALF_SCALE + [.05], None, "half",
    [0.] + HALF_SCALE[1:], [-.015] + HALF_SCALE[1:],
    [float("nan")] + HALF_SCALE[1:], [float("inf")] + HALF_SCALE[1:],
    [True] + HALF_SCALE[1:], [".015"] + HALF_SCALE[1:], [None] + HALF_SCALE[1:],
])
def test_supervisor_rejects_invalid_residual_scale(tmp_path, scale):
    selected = half_scale_config()
    selected["learner"]["residual_scale"] = scale
    with pytest.raises(ValueError, match="residual_scale"):
        load_config(tmp_path, selected)


@pytest.mark.parametrize("field,value", [
    (("learner", "noise_scale"), .75),
    (("learner", "policy_lr"), 2e-4),
    (("learner", "batch_size"), 128),
    (("progressive_exploration",), 10_000),
    (("max_transitions",), 1_000_000),
    (("seed",), 43),
    (("resume",), True),
])
def test_supervisor_rejects_unrequested_hyperparameter_changes(tmp_path, field, value):
    selected = half_scale_config()
    target = selected
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = value
    with pytest.raises(ValueError):
        load_config(tmp_path, selected)


def test_half_scale_matches_between_critic_and_executed_actions_without_scaling_noise(prepared_runner):
    runner, decoder, items = prepared_runner
    runner.learner = SACLearner(LearnerConfig(
        method="action_residual", obs_dim=runner.obs_dim, hidden_dim=8,
        residual_scale=HALF_SCALE), device="cpu")
    prepared = runner._encode(items)
    # Exercise every axis and every horizon element, including the two bounds.
    normalized = np.linspace(-1, 1, 2 * 7 * 9, dtype=np.float32).reshape(2, 63)
    runner.learner.act = lambda *args, **kwargs: normalized.copy()
    actions, replay_actions = runner._actions(prepared)
    base = np.stack([prepared[eid][1]["base_actions_pose9"] for eid in sorted(prepared)])
    expected = base + normalized.reshape(2, 7, 9) * np.asarray(HALF_SCALE)
    np.testing.assert_allclose([row["actions_pose8"] for row in actions], pose9_to_pose8(expected),
                               atol=1e-12)
    critic = runner.learner._critic_action(torch.from_numpy(normalized),
                                          torch.as_tensor(base.reshape(2, 63), dtype=torch.float32))
    np.testing.assert_allclose(critic.numpy().reshape(2, 7, 9), expected, atol=1e-7)
    np.testing.assert_array_equal(np.stack([replay_actions[eid] for eid in sorted(prepared)]), normalized)
    # Frozen noise decode stays at .125 * 1.5, not the halved residual magnitude.
    assert len(runner.noise_policy.calls) == 1 and runner.noise_policy.config.noise_scale == 1.5
    assert [path for path, _ in decoder.requests] == ["/encode", "/decode"]
    for item in decoder.requests[-1][1]["items"]:
        np.testing.assert_array_equal(item["initial_noise"], np.full((7, 9), .1875))


def test_half_scale_fresh_training_updates_residual_only(tmp_path, monkeypatch):
    simulation, inference = GammaIndexedSimulation(.995), BatchInference()
    noise_path = tmp_path / "noise.pt"
    cfg = config(max_transitions=5, learning_start=2, pipeline_updates=False, single_gamma=.995)
    cfg.learner["residual_scale"] = HALF_SCALE
    monkeypatch.setattr(SACLearner, "load", lambda *args, **kwargs:
                        pytest.fail("Fresh half-scale training must not load a residual checkpoint"))
    with serve(simulation.handlers) as sim_url, serve(inference.handlers) as infer_url:
        runner = ResidualOnNoiseRunner(cfg, sim_url, infer_url, tmp_path / "run",
            noise_checkpoint=noise_path, device="cpu", token=TOKEN, timeout=5)
        try:
            runner._verify_nodes()
            source = deepcopy(runner.identities)
            source["simulation"]["single_gamma"] = .99
            noise_payload, frozen_hash = noise_checkpoint(noise_path, obs_dim=runner.obs_dim, identities=source)
            checkpoint_hash = file_sha256(noise_path)
            result = runner.run()
        finally:
            runner.executor.shutdown(wait=True, cancel_futures=True)
    assert result["state"] == "complete"
    assert result["training_transitions"] == 5 and result["updates"] == 3
    assert runner.checkpoint is None and not runner.resume
    assert runner.learner.config.residual_scale == tuple(HALF_SCALE)
    assert runner.noise_policy.config.noise_scale == noise_payload["config"]["noise_scale"] == 1.5
    assert runner.noise_policy.config.residual_scale == tuple(noise_payload["config"]["residual_scale"])
    assert runner.noise_policy.verify_frozen() == frozen_hash
    assert file_sha256(noise_path) == checkpoint_hash
    initial = torch.load(runner.output / "initial.pt", map_location="cpu", weights_only=False)
    assert initial["config"]["residual_scale"] == tuple(HALF_SCALE)
    assert initial["updates"] == initial["extra"]["replay_size"] == 0
    assert all(not initial[key]["state"] for key in ("actor_optimizer", "q_optimizer", "alpha_optimizer"))
    assert any(not torch.equal(initial["actor"][name], value)
               for name, value in runner.learner.actor.state_dict().items())
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in runner.noise_policy.actor.parameters())
    composition = json.loads((runner.output / "composition.json").read_text())
    assert composition["residual_initialization"] == "from scratch"
    assert composition["frozen_noise"]["actor_sha256"] == frozen_hash
    assert composition["frozen_noise"]["noise_scale"] == 1.5
    assert composition["frozen_noise_objective_transfer"] == {
        "source_single_gamma": .99, "runtime_single_gamma": .995, "changed": True}
