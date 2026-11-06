"""SAC boundary checks for non-elevator action and latent-noise schemas."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pressb.online_rl.learner import LearnerConfig, ReplayBuffer, SACLearner


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def batch_for(learner, count=3):
    cfg = learner.config
    rng = np.random.default_rng(17)
    batch = {
        "obs": rng.normal(size=(count, cfg.obs_dim)).astype(np.float32),
        "next_obs": rng.normal(size=(count, cfg.obs_dim)).astype(np.float32),
        "action": rng.uniform(-1, 1, (count, cfg.action_dim)).astype(np.float32),
        "reward": np.array([[1.], [0.], [-.1]], np.float32)[:count],
        "discount": np.full((count, 1), .99, np.float32),
        "terminated": np.array([[True], [False], [False]])[:count],
        "truncated": np.array([[False], [True], [False]])[:count],
    }
    if learner.residual:
        batch.update({name: rng.normal(size=(count, *cfg.base_action_shape)).astype(np.float32)
                      for name in ("base_action", "next_base_action")})
    return batch


@pytest.mark.parametrize("method", ["initial_noise", "action_residual"])
def test_pap_shapes_train_and_resume_exactly(tmp_path, method):
    cfg = LearnerConfig(method=method, obs_dim=11, hidden_dim=8, batch_size=3,
                        action_horizon=32, model_action_dim=7, noise_dim=20,
                        noise_steps=32, residual_mode="all", residual_scale=(.02,) * 7,
                        residual_q_bounds=None, policy_freq=1, target_freq=1,
                        action_lower=(-1.,) * 7, action_upper=(1.,) * 7)
    learner = SACLearner(cfg)
    assert learner.action_dim == (640 if method == "initial_noise" else 224)
    batch = batch_for(learner)
    first = learner.update(batch)
    assert first["actor_updated"] == 1
    assert all(np.isfinite(x) for x in first.values())
    replay = ReplayBuffer(4, cfg.obs_dim, cfg.action_dim, base_action_shape=cfg.base_action_shape)
    for i in range(3):
        replay.add(batch["obs"][i], batch["action"][i], float(batch["reward"][i, 0]),
                   float(batch["discount"][i, 0]), batch["next_obs"][i],
                   batch["terminated"][i, 0], batch["truncated"][i, 0],
                   batch["base_action"][i] if learner.residual else None,
                   batch["next_base_action"][i] if learner.residual else None)
    path = tmp_path / "checkpoint.pt"
    learner.save(path, extra={"replay": replay.state_dict()})
    expected_action = learner.act(batch["obs"], batch.get("base_action"))
    expected_update = learner.update(batch)
    restored = SACLearner.load(path)
    assert restored.config.base_action_shape == (32, 7)
    assert restored.config.noise_dim == 20
    assert restored.config.action_lower == (-1.,) * 7
    np.testing.assert_array_equal(restored.act(batch["obs"], batch.get("base_action")), expected_action)
    assert restored.update(batch) == expected_update
    restored_replay = ReplayBuffer(4, cfg.obs_dim, cfg.action_dim,
                                    base_action_shape=(32, 7))
    restored_replay.load_state_dict(restored.checkpoint_extra["replay"])
    # Sample once per buffer so the sampled rows and RNG continuation agree.
    expected_batch, actual_batch = replay.sample(5), restored_replay.sample(5)
    for key in expected_batch:
        np.testing.assert_array_equal(expected_batch[key], actual_batch[key])


def test_subset_residual_critic_preserves_gripper_and_untouched_joints():
    cfg = LearnerConfig(method="action_residual", obs_dim=4, hidden_dim=8,
                        action_horizon=32, model_action_dim=7, residual_mode="all",
                        residual_indices=(5, 1, 3), residual_scale=(.1, .2, .3))
    learner = SACLearner(cfg)
    base = torch.arange(224, dtype=torch.float32).reshape(1, -1)
    action = torch.ones((1, 96), requires_grad=True)
    combined = learner._critic_action(action, base).reshape(1, 32, 7)
    expected = base.reshape(1, 32, 7).clone()
    expected[..., 5] += .1
    expected[..., 1] += .2
    expected[..., 3] += .3
    torch.testing.assert_close(combined, expected)
    torch.testing.assert_close(combined[..., (0, 2, 4, 6)], base.reshape(1, 32, 7)[..., (0, 2, 4, 6)])
    combined.sum().backward()
    torch.testing.assert_close(action.grad, torch.tensor([.1, .2, .3]).repeat(32).reshape(1, 96))
    batch = batch_for(learner)
    learner.update(batch)
    assert learner.update(batch)["actor_updated"] == 1


def test_residual_projection_matches_executable_clip_and_gradients():
    learner = SACLearner(LearnerConfig(method="action_residual", obs_dim=4, hidden_dim=8,
                        action_horizon=2, model_action_dim=3, residual_mode="all",
                        residual_indices=(0, 2), residual_scale=(.2, .1),
                        action_lower=(-1., 0., 0.), action_upper=(1., 1., 1.), policy_freq=1))
    base = torch.tensor([[.95, -2., .5, 0., .3, .99]])
    action = torch.tensor([[1., 1., -.5, 1.]], requires_grad=True)
    combined = learner._critic_action(action, base)
    torch.testing.assert_close(combined, torch.tensor([[1., 0., .6, -.1, .3, 1.]]))
    combined.sum().backward()
    torch.testing.assert_close(action.grad, torch.tensor([[0., .1, .2, 0.]]))
    evaluated = []
    handles = [model.register_forward_pre_hook(lambda model, inputs: evaluated.append(inputs[1].detach().clone()))
               for model in (learner.qs, learner.q_targets)]
    try:
        learner.update(batch_for(learner))
    finally:
        for handle in handles:
            handle.remove()
    # Current-action critic, Bellman target critic, and policy critic all use
    # the executable commands, including unchanged coordinates of the base.
    assert len(evaluated) == 3
    for command in evaluated:
        assert torch.all(command >= learner.action_lower)
        assert torch.all(command <= learner.action_upper)


def test_action_bounds_do_not_clip_the_dsrl_latent_critic_action():
    learner = SACLearner(LearnerConfig(method="initial_noise", obs_dim=4, hidden_dim=8,
                        action_horizon=32, model_action_dim=7, noise_dim=20,
                        action_lower=(0.,) * 7, action_upper=(.1,) * 7))
    latent = torch.linspace(-1., 1., 20).reshape(1, 20)
    assert learner._critic_action(latent, None) is latent


@pytest.mark.parametrize("bounds, expected", [(None, 3.), ((-2., 4.), -2. + 3. * (np.tanh(3.) + 1.))])
def test_residual_q_can_model_dense_or_signed_return(bounds, expected):
    learner = SACLearner(LearnerConfig(method="action_residual", obs_dim=4, hidden_dim=8,
                                       residual_q_bounds=bounds))
    with torch.no_grad():
        learner.qs.net[-1].weight.zero_()
        learner.qs.net[-1].bias.fill_(3.)
        values = learner.qs(torch.zeros(2, 4), torch.zeros(2, 63))
    torch.testing.assert_close(values, torch.full_like(values, expected))


def test_single_step_caps_has_zero_temporal_penalty():
    learner = SACLearner(LearnerConfig(method="action_residual", obs_dim=4, hidden_dim=8,
                                       action_horizon=1, model_action_dim=2,
                                       residual_mode="all", residual_scale=(.1, .1),
                                       lambda_s=1., lambda_t=1., policy_freq=1))
    result = learner.update(batch_for(learner))
    assert result["temporal_smoothness_loss"] == 0.
    assert all(np.isfinite(x) for x in result.values())


def test_original_checkpoint_and_replay_without_schema_fields_load(tmp_path):
    learner = SACLearner(LearnerConfig(method="action_residual", obs_dim=4, hidden_dim=8))
    path = tmp_path / "legacy.pt"
    learner.save(path)
    payload = torch.load(path, weights_only=False)
    for name in ("action_horizon", "model_action_dim", "noise_dim", "residual_indices", "residual_q_bounds",
                 "action_lower", "action_upper"):
        payload["config"].pop(name)
    torch.save(payload, path)
    restored = SACLearner.load(path)
    obs, base = np.zeros((1, 4)), np.zeros((1, 7, 9))
    np.testing.assert_array_equal(learner.act(obs, base, deterministic=True),
                                  restored.act(obs, base, deterministic=True))
    replay = ReplayBuffer(2, 4, 63)
    replay.add(obs[0], np.zeros(63), 0, .9, obs[0], False, False, base[0], base[0])
    state = replay.state_dict()
    state.pop("base_action_shape")
    ReplayBuffer(2, 4, 63).load_state_dict(state)
    with pytest.raises(ValueError, match="dimensions"):
        ReplayBuffer(2, 4, 63, base_action_shape=(9, 7)).load_state_dict(state)


@pytest.mark.parametrize("overrides", [
    {"action_horizon": 0}, {"model_action_dim": True}, {"noise_dim": -1},
    {"action_horizon": 32, "noise_steps": 3}, {"residual_indices": ()},
    {"residual_indices": (0, 0)}, {"residual_indices": (0, 9)},
    {"residual_indices": (False, 1)}, {"residual_mode": "xyz", "model_action_dim": 2},
    {"residual_mode": "xyz", "residual_indices": (1, 2, 3)},
    {"residual_q_bounds": (1., 0.)}, {"residual_q_bounds": (0., float("inf"))},
    {"action_lower": (0.,) * 9}, {"action_upper": (1.,) * 9},
    {"action_lower": (0.,) * 8, "action_upper": (1.,) * 9},
    {"action_lower": (False,) * 9, "action_upper": (1.,) * 9},
    {"action_lower": (-float("inf"),) * 9, "action_upper": (1.,) * 9},
    {"action_lower": (1.,) * 9, "action_upper": (1.,) * 9},
])
def test_invalid_generic_contract_rejected(overrides):
    with pytest.raises(ValueError):
        LearnerConfig(method="action_residual", obs_dim=4, **overrides)
