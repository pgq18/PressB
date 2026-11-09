"""Critic-only warmup and actor-only KL stopping; all execution stays on CPU."""
import json

import pytest
import torch

from pressb.runtime.dppo import DPPOConfig
from pressb.runtime.dppo_trainer import _algorithm_identity
from pressb.runtime.expert_sync import state_sha256
from test_dppo_training import make_runner


def records(path, name='events.jsonl'):
    return [json.loads(line) for line in (path / name).read_text().splitlines()]


@pytest.mark.parametrize('value', [-1, True, 1.5, '2', None])
def test_warmup_count_rejects_invalid_counts(value):
    with pytest.raises(ValueError, match='critic_warmup_iterations'):
        DPPOConfig(critic_warmup_iterations=value)


@pytest.mark.parametrize('value', [0, 1, None, 'true'])
def test_critic_continue_flag_requires_boolean(value):
    with pytest.raises(ValueError, match='critic_continue_after_kl'):
        DPPOConfig(critic_continue_after_kl=value)


def test_warmup_never_builds_actor_grad_or_computes_kl_and_keeps_fresh_rollouts(tmp_path, monkeypatch):
    runner, client, _ = make_runner(tmp_path, steps=8)
    runner.config.dppo.critic_warmup_iterations = 2
    runner.config.dppo.target_kl = 1e-30
    initial_actor = state_sha256(runner.expert.state_dict())
    initial_critic = []
    original_negotiate = runner._negotiate
    def negotiate():
        original_negotiate()
        initial_critic.append(state_sha256(runner.value.state_dict()))
    monkeypatch.setattr(runner, '_negotiate', negotiate)
    monkeypatch.setattr('pressb.runtime.dppo_trainer.clipped_policy_loss',
                        lambda *args, **kwargs: pytest.fail('Warmup must not compute actor loss or KL'))
    grad_modes = []
    original_transition = runner.expert.transition
    def transition(*args, **kwargs):
        grad_modes.append(torch.is_grad_enabled())
        return original_transition(*args, **kwargs)
    monkeypatch.setattr(runner.expert, 'transition', transition)
    result = runner.run()
    assert runner.iterations == 2
    assert runner.actor_updates == 0 and runner.critic_updates == runner.updates == 8
    assert result['critic_warmup_iterations_remaining'] == 0
    assert state_sha256(runner.expert.state_dict()) == initial_actor
    assert state_sha256(runner.value.state_dict()) != initial_critic[0]
    assert all(p.grad is None for p in runner.expert.parameters())
    assert grad_modes and not any(grad_modes)
    assert not runner.actor_optimizer.optimizer.state
    assert runner.policy_version == 0
    assert not [route for route, _ in client.calls if route.startswith('/weights_')]
    sample_requests = [body for route, body in client.calls if route == '/sample_chain']
    assert len(sample_requests) == 4 and len({tuple(body['seeds']) for body in sample_requests}) == 4
    event = [row for row in records(tmp_path) if row['event'] == 'iteration_optimized']
    assert all(row['critic_warmup'] and row['actor_minibatches_accepted'] == 0
               and row['actor_minibatches_rejected'] == 0 and row['rejected_approx_kl'] is None
               and row['actor_minibatches_skipped'] == row['planned_minibatches'] == 4 for row in event)
    assert len([row for row in records(tmp_path) if row['event'] == 'rollout_logprob_verified']) == 2
    assert all(row['nonzero_reward_count'] == 2 and row['raw_advantage_std'] > 0
               and row['return_min'] < row['return_max'] == 1 for row in event)


def test_warmup_checkpoint_resume_finishes_remaining_iteration_then_updates_actor(tmp_path):
    runner, _, _ = make_runner(tmp_path / 'first', steps=4)
    runner.config.dppo.critic_warmup_iterations = 2
    first = runner.run()
    payload = torch.load(first['last_checkpoint'], weights_only=True)
    assert payload['iterations'] == 1 and payload['actor_updates'] == 0 and payload['critic_updates'] == 4
    initial_actor = state_sha256(payload['expert'])
    resumed, client, _ = make_runner(tmp_path / 'resumed', steps=12, checkpoint=first['last_checkpoint'])
    resumed.config.dppo.critic_warmup_iterations = 2
    result = resumed.run()
    assert resumed.iterations == 3 and resumed.actor_updates == 4 and resumed.critic_updates == 12
    assert state_sha256(resumed.expert.state_dict()) != initial_actor
    phases = [row['critic_warmup'] for row in records(tmp_path / 'resumed') if row['event'] == 'iteration_optimized']
    assert phases == [True, False]
    calls = [body['expected_policy_version'] for route, body in client.calls if route == '/sample_chain']
    assert calls == [1, 1, 1, 1]  # restore publishes once; warmup does not publish
    final = torch.load(result['last_checkpoint'], weights_only=True)
    assert final['iterations'] == 3 and final['actor_updates'] == 4 and final['critic_updates'] == 12
    assert final['policy_version'] == 2


def test_evaluation_has_no_pending_critic_training(tmp_path):
    runner, _, _ = make_runner(tmp_path, mode='eval', steps=40)
    runner.config.dppo.critic_warmup_iterations = 2
    result = runner.run()
    assert result['state'] == 'completed'
    assert result['critic_warmup_iterations_remaining'] is None
    assert result['actor_updates'] == result['critic_updates'] == 0


@pytest.mark.parametrize('continue_critic,expected_critic', [(False, 1), (True, 4)])
def test_kl_stop_records_rejected_kl_and_optionally_finishes_critic(tmp_path, monkeypatch, continue_critic, expected_critic):
    runner, _, _ = make_runner(tmp_path, steps=4)
    runner.config.dppo.target_kl = .05
    runner.config.dppo.critic_continue_after_kl = continue_critic
    import pressb.runtime.dppo_trainer as module
    original = module.clipped_policy_loss
    calls = []
    def policy_loss(*args, **kwargs):
        loss, metrics = original(*args, **kwargs)
        calls.append(None)
        metrics['approx_kl'] = .001 if len(calls) == 1 else .75
        return loss, metrics
    monkeypatch.setattr(module, 'clipped_policy_loss', policy_loss)
    runner.run()
    assert len(calls) == 2  # no further actor graph after rejected KL
    assert runner.actor_updates == 1 and runner.critic_updates == expected_critic
    metrics = runner.latest_metrics
    assert metrics['early_stop_kl'] and metrics['rejected_approx_kl'] == .75
    assert metrics['approx_kl'] == .001  # accepted statistic is explicitly distinct
    assert metrics['actor_minibatches_accepted'] == 1
    assert metrics['actor_minibatches_rejected'] == 1
    assert metrics['actor_minibatches_skipped'] == 2
    assert metrics['critic_minibatches'] == expected_critic
    assert metrics['planned_minibatches'] == 4


def test_default_identity_preserves_legacy_checkpoint_and_nonzero_option_rejects(tmp_path):
    cfg = DPPOConfig()
    identity = _algorithm_identity(cfg)
    assert 'critic_warmup_iterations' not in identity
    assert 'critic_continue_after_kl' not in identity
    runner, _, _ = make_runner(tmp_path / 'first', steps=4)
    saved = runner.run()['last_checkpoint']
    payload = torch.load(saved, weights_only=True)
    # Strip fields absent from an old-format checkpoint. Its original algorithm
    # identity already omits the two default-disabled fields.
    payload.pop('actor_updates'); payload.pop('critic_updates')
    payload['config']['dppo'].pop('critic_warmup_iterations')
    payload['config']['dppo'].pop('critic_continue_after_kl')
    legacy = tmp_path / 'legacy.pt'
    torch.save(payload, legacy)
    resumed, _, _ = make_runner(tmp_path / 'resumed', steps=8, checkpoint=legacy)
    resumed.run()
    assert resumed.actor_updates == resumed.critic_updates == resumed.updates == 8
    changed, _, _ = make_runner(tmp_path / 'changed', steps=8, checkpoint=legacy)
    changed.config.dppo.critic_warmup_iterations = 1
    with pytest.raises(ValueError, match='identity differs'):
        changed.run()
    changed_flag, _, _ = make_runner(tmp_path / 'changed_flag', steps=8, checkpoint=legacy)
    changed_flag.config.dppo.critic_continue_after_kl = True
    with pytest.raises(ValueError, match='identity differs'):
        changed_flag.run()
