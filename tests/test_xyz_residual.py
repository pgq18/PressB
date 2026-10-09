"""CPU contracts for learning 7xXYZ residuals on an immutable noise policy."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pressb.online_rl.learner import LearnerConfig, SACLearner
from pressb.online_rl.protocol import residual_actions
from pressb.online_rl.residual_on_noise import (
    ResidualOnNoiseRunner, actor_sha256, file_sha256,
)
from test_fast_online_rl_runner import BatchInference, config
from test_online_rl_learner import make_batch
from test_online_rl_runner import TOKEN, serve
from test_residual_on_noise import GammaIndexedSimulation, noise_checkpoint, prepared_runner


ROOT = Path(__file__).resolve().parents[1]
XYZ_SCALE = [.02, .02, .02]
CONFIG = ROOT / "configs/online_rl_residual_on_noise_gamma0995_xyz002_400k.json"
spec = importlib.util.spec_from_file_location(
    "xyz_residual_experiment_supervisor", ROOT / "scripts/run_residual_on_noise_experiment.py")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def xyz_learner(**changes):
    values = dict(method="action_residual", obs_dim=4, hidden_dim=8, batch_size=4,
                  residual_mode="xyz", residual_scale=XYZ_SCALE)
    values.update(changes)
    return SACLearner(LearnerConfig(**values), device="cpu")


def experiment_config(tmp_path, value):
    path = tmp_path / "selected.json"
    path.write_text(json.dumps(value))
    return supervisor.Experiment(tmp_path / "run", tmp_path / "noise.pt", config=path).load_config()


def test_xyz_experiment_changes_only_mode_and_scale_from_gamma0995(tmp_path):
    selected = json.loads(CONFIG.read_text())
    expected = json.loads((ROOT / "configs/online_rl_residual_on_noise_gamma0995_400k.json").read_text())
    expected["learner"].update(residual_mode="xyz", residual_scale=XYZ_SCALE)
    assert selected == expected
    assert selected["max_transitions"] == 400_000 and selected["single_gamma"] == .995
    assert experiment_config(tmp_path, selected) == selected


@pytest.mark.parametrize("scale", [
    [], [.02] * 2, [.02] * 4, [.02] * 9, None, ".02",
    [0., .02, .02], [-.02, .02, .02], [float("nan"), .02, .02],
    [float("inf"), .02, .02], [True, .02, .02], [".02", .02, .02], [None, .02, .02],
])
def test_xyz_scale_requires_exactly_three_finite_positive_numbers(tmp_path, scale):
    with pytest.raises(ValueError, match="residual_scale"):
        LearnerConfig(method="action_residual", obs_dim=4, residual_mode="xyz", residual_scale=scale)
    selected = json.loads(CONFIG.read_text())
    selected["learner"]["residual_scale"] = scale
    with pytest.raises(ValueError, match="residual_scale"):
        experiment_config(tmp_path, selected)


@pytest.mark.parametrize("mode", ["rotation", "XYZ", "", None, 3, True])
def test_unknown_residual_modes_are_rejected(tmp_path, mode):
    with pytest.raises(ValueError, match="residual_mode"):
        LearnerConfig(method="action_residual", obs_dim=4, residual_mode=mode, residual_scale=XYZ_SCALE)
    selected = json.loads(CONFIG.read_text())
    selected["learner"]["residual_mode"] = mode
    with pytest.raises(ValueError, match="residual_mode"):
        experiment_config(tmp_path, selected)


def test_xyz_cannot_shrink_frozen_initial_noise_policy():
    with pytest.raises(ValueError):
        LearnerConfig(method="initial_noise", obs_dim=4, residual_mode="xyz", residual_scale=XYZ_SCALE)
    noise = LearnerConfig(method="initial_noise", obs_dim=4)
    old_residual = LearnerConfig(method="action_residual", obs_dim=4)
    assert noise.action_dim == 9 and noise.residual_mode == "pose9"
    assert old_residual.action_dim == 63 and old_residual.residual_mode == "pose9"


@pytest.mark.parametrize("field,value", [
    (("learner", "noise_scale"), .75), (("learner", "policy_lr"), 2e-4),
    (("learner", "target_entropy"), -31.5), (("max_transitions",), 1_000_000),
    (("resume",), True),
])
def test_xyz_experiment_rejects_unrequested_changes(tmp_path, field, value):
    selected = json.loads(CONFIG.read_text())
    target = selected
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = value
    with pytest.raises(ValueError):
        experiment_config(tmp_path, selected)


def test_xyz_actor_is_21d_but_conditions_and_critics_use_the_complete_base_pose():
    learner = xyz_learner()
    obs = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    base = torch.arange(189, dtype=torch.float32).reshape(3, 63) / 189
    assert learner.action_dim == learner.actor.mean.out_features == learner.actor.log_std.out_features == 21
    assert learner.actor.net[0].in_features == 4 + 63
    assert learner.target_entropy == -10.5
    actor_input = learner._actor_input(obs, base)
    torch.testing.assert_close(actor_input[:, 4:], base, rtol=0, atol=0)
    output, log_prob, _ = learner.actor.sample(actor_input)
    assert output.shape == (3, 21) and log_prob.shape == (3, 1)
    assert torch.isfinite(output).all() and (output.abs() <= 1).all()
    values = learner.qs(obs, learner._critic_action(output, base))
    assert values.shape == (2, 3, 1) and torch.isfinite(values).all()
    assert (values >= 0).all() and (values <= 1).all()


def test_xyz_combination_preserves_rotations_and_maps_all_horizon_axis_gradients():
    learner = xyz_learner()
    base = torch.arange(126, dtype=torch.float32).reshape(2, 63) / 126
    original = base.clone()
    action = torch.linspace(-1, 1, 42).reshape(2, 21).requires_grad_()
    combined = learner._critic_action(action, base).reshape(2, 7, 9)
    torch.testing.assert_close(combined[:, :, 3:], base.reshape(2, 7, 9)[:, :, 3:], rtol=0, atol=0)
    torch.testing.assert_close(combined[:, :, :3] - base.reshape(2, 7, 9)[:, :, :3],
                               action.reshape(2, 7, 3) * .02, atol=3e-8, rtol=1e-5)
    torch.testing.assert_close(base, original, rtol=0, atol=0)
    assert (combined[:, :, :3] - base.reshape(2, 7, 9)[:, :, :3]).abs().max() <= .0200001
    # Distinct weights expose transpositions, wrong stride, detached insertion,
    # and gradients accidentally passing through any of the six rotation axes.
    weights = torch.arange(1, 127, dtype=torch.float32).reshape(2, 7, 9)
    (combined * weights).sum().backward()
    torch.testing.assert_close(action.grad.reshape(2, 7, 3), weights[:, :, :3] * .02)
    for row in range(2):
        decoded = residual_actions(base[row].numpy().reshape(7, 9), action[row].detach().numpy(),
                                   XYZ_SCALE, residual_mode="xyz")
        np.testing.assert_allclose(decoded, combined[row].detach().numpy(), atol=6e-8)
        np.testing.assert_array_equal(decoded[:, 3:], original[row].numpy().reshape(7, 9)[:, 3:])


@pytest.mark.parametrize("warmup", [False, True])
def test_xyz_rollout_retains_exact_decoder_quaternion_and_gripper(prepared_runner, warmup):
    runner, decoder, items = prepared_runner
    runner.learner = xyz_learner(obs_dim=runner.obs_dim)
    prepared = runner._encode(items)
    # Preserve a non-default gripper too: conversion from pose9 would lose it.
    for row in prepared.values():
        for step, pose in enumerate(row[1]["base_actions_pose8"]):
            pose[7] = .012 + step * .0001
    original = deepcopy(prepared)
    normalized = (np.zeros((2, 21), np.float32) if warmup else
                  np.linspace(-1, 1, 42, dtype=np.float32).reshape(2, 21))
    if not warmup:
        runner.learner.act = lambda *args, **kwargs: normalized.copy()
    actions, replay = runner._actions(prepared)
    assert [row["env_id"] for row in actions] == sorted(prepared)
    for index, row in enumerate(actions):
        eid = row["env_id"]
        wire = np.asarray(original[eid][1]["base_actions_pose8"])
        actual = np.asarray(row["actions_pose8"])
        np.testing.assert_array_equal(actual[:, 3:], wire[:, 3:])
        if warmup:
            np.testing.assert_array_equal(actual, wire)
        else:
            raw = np.asarray(original[eid][1]["base_actions_pose9"])
            np.testing.assert_allclose(actual[:, :3], raw[:, :3] + normalized[index].reshape(7, 3) * .02,
                                       rtol=0, atol=2e-9)
        np.testing.assert_array_equal(replay[eid], normalized[index])
        np.testing.assert_array_equal(prepared[eid][1]["base_actions_pose8"], wire)
    assert len(runner.noise_policy.calls) == 1
    assert [name for name, _ in decoder.requests] == ["/encode", "/decode"]
    for item in decoder.requests[-1][1]["items"]:
        np.testing.assert_array_equal(item["initial_noise"], np.full((7, 9), .1875))


def test_xyz_sac_update_and_checkpoint_roundtrip_preserve_21d_optimizers_and_rng(tmp_path):
    learner = xyz_learner(policy_freq=1, target_freq=1)
    batch = make_batch(learner)
    before = {key: deepcopy(getattr(learner, key).state_dict()) for key in ("actor", "qs")}
    metrics = learner.update(batch)
    assert metrics["actor_updated"] == metrics["target_updated"] == 1
    assert all(np.isfinite(value) for value in metrics.values())
    for key in before:
        assert any(not torch.equal(value, getattr(learner, key).state_dict()[name])
                   for name, value in before[key].items())
    assert learner.actor_optimizer.state and learner.q_optimizer.state and learner.alpha_optimizer.state
    checkpoint = tmp_path / "xyz.pt"
    learner.save(checkpoint)
    expected_action = learner.act(batch["obs"], batch["base_action"], exploration_probability=.5)
    expected_metrics = learner.update(batch)
    restored = SACLearner.load(checkpoint, device="cpu")
    assert restored.action_dim == 21 and restored.config.residual_mode == "xyz"
    assert restored.target_entropy == -10.5
    np.testing.assert_array_equal(restored.act(batch["obs"], batch["base_action"], exploration_probability=.5),
                                  expected_action)
    assert restored.update(batch) == expected_metrics
    for key in ("actor", "qs", "q_targets"):
        for name, value in getattr(learner, key).state_dict().items():
            torch.testing.assert_close(value, getattr(restored, key).state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("method", ["action_residual", "initial_noise"])
def test_legacy_checkpoints_without_residual_mode_keep_original_networks(tmp_path, method):
    learner = SACLearner(LearnerConfig(method=method, obs_dim=4, hidden_dim=8, batch_size=4), device="cpu")
    batch = make_batch(learner)
    learner.update(batch)
    checkpoint = tmp_path / "legacy.pt"
    learner.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["config"].pop("residual_mode")
    torch.save(payload, checkpoint)
    expected = learner.act(batch["obs"], batch.get("base_action"), deterministic=True)
    restored = SACLearner.load(checkpoint, device="cpu")
    assert restored.config.residual_mode == "pose9"
    assert restored.action_dim == (63 if method == "action_residual" else 9)
    np.testing.assert_array_equal(restored.act(batch["obs"], batch.get("base_action"), deterministic=True), expected)


@pytest.mark.parametrize("pipeline", [False, True])
def test_xyz_fresh_http_training_and_frozen_evaluation(tmp_path, monkeypatch, pipeline):
    noise_path = tmp_path / "noise.pt"
    train_cfg = config(max_transitions=5, learning_start=2, pipeline_updates=pipeline, single_gamma=.995)
    train_cfg.learner.update(residual_mode="xyz", residual_scale=XYZ_SCALE)
    simulation, inference = GammaIndexedSimulation(.995), BatchInference()
    with serve(simulation.handlers) as sim_url, serve(inference.handlers) as infer_url:
        trained = ResidualOnNoiseRunner(train_cfg, sim_url, infer_url, tmp_path / "train",
            noise_checkpoint=noise_path, device="cpu", token=TOKEN, timeout=5)
        try:
            trained._verify_nodes()
            source = deepcopy(trained.identities)
            source["simulation"]["single_gamma"] = .99
            payload, frozen_hash = noise_checkpoint(noise_path, obs_dim=trained.obs_dim, identities=source)
            # The actual immutable learned-noise checkpoint predates this mode.
            payload["config"].pop("residual_mode", None)
            torch.save(payload, noise_path)
            checkpoint_hash = file_sha256(noise_path)
            with monkeypatch.context() as fresh:
                fresh.setattr(SACLearner, "load", lambda *args, **kwargs:
                              pytest.fail("Fresh XYZ training must not restore a residual checkpoint"))
                result = trained.run()
        finally:
            trained.executor.shutdown(wait=True, cancel_futures=True)
    assert result["state"] == "complete" and result["training_transitions"] == 5
    assert result["updates"] == 3 and trained.learner.action_dim == 21
    assert not trained.resume and trained.checkpoint is None
    replay = trained.replay.state_dict()["arrays"]
    assert replay["action"].shape == (5, 21) and replay["base_action"].shape == (5, 63)
    np.testing.assert_array_equal(replay["action"][:2], 0.)
    np.testing.assert_array_equal(replay["next_base_action"][1], replay["base_action"][3])
    initial = torch.load(trained.output / "initial.pt", map_location="cpu", weights_only=False)
    assert initial["updates"] == initial["extra"]["replay_size"] == 0
    assert initial["config"]["residual_mode"] == "xyz"
    assert initial["config"]["residual_scale"] == tuple(XYZ_SCALE)
    assert all(not initial[name]["state"] for name in ("actor_optimizer", "q_optimizer", "alpha_optimizer"))
    assert any(not torch.equal(value, trained.learner.actor.state_dict()[name])
               for name, value in initial["actor"].items())
    assert trained.noise_policy.verify_frozen() == frozen_hash
    assert trained.noise_policy.config.action_dim == 9 and trained.noise_policy.config.noise_scale == 1.5
    assert file_sha256(noise_path) == checkpoint_hash
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in trained.noise_policy.actor.parameters())
    trained_actor = actor_sha256(trained.learner.actor)
    eval_cfg = deepcopy(train_cfg)
    eval_cfg.mode, eval_cfg.eval_episodes = "eval", 2
    simulation, inference = GammaIndexedSimulation(.995), BatchInference()
    with serve(simulation.handlers) as sim_url, serve(inference.handlers) as infer_url:
        evaluated = ResidualOnNoiseRunner(eval_cfg, sim_url, infer_url, tmp_path / "eval",
            noise_checkpoint=noise_path, checkpoint=trained.output / "last.pt",
            device="cpu", token=TOKEN, timeout=5)
        try:
            evaluated_result = evaluated.run()
        finally:
            evaluated.executor.shutdown(wait=True, cancel_futures=True)
    assert evaluated_result["state"] == "complete" and evaluated_result["episodes"] == 2
    assert evaluated.learner.updates == 3 and len(evaluated.replay) == evaluated.earned_update_credit == 0
    assert evaluated.learner.action_dim == 21 and actor_sha256(evaluated.learner.actor) == trained_actor
    assert evaluated.noise_policy.verify_frozen() == frozen_hash and file_sha256(noise_path) == checkpoint_hash
    assert all(not parameter.requires_grad for module in
               (evaluated.learner.actor, evaluated.learner.qs, evaluated.learner.q_targets)
               for parameter in module.parameters())
    assert not inference.contexts and not evaluated.contexts
