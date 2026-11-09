"""DPPO math plus the real inference/update wire protocol on CPU toy tasks."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from torch import nn

from pressb.runtime.dppo import (DPPOConfig, generalized_advantages, reduce_logprobs,
                                 denoising_clip, clipped_policy_loss, clipped_value_loss)
from pressb.runtime.dppo_trainer import DPPOTrainer, DPPOTrainerConfig, MasterAdamW
from pressb.runtime.expert_sync import state_sha256
from pressb.runtime.inference import InferenceService


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_gae_terminal_truncation_and_actual_chunk_discounts():
    reward = torch.tensor([1., 2., 100.])
    value = torch.tensor([.5, .7, .9])
    next_value = torch.tensor([.7, 3., 10.])
    discounts = torch.tensor([.9, .81, .5])
    terminated = torch.tensor([False, False, True])
    truncated = torch.tensor([False, True, False])
    advantage, returns = generalized_advantages(reward, value, next_value, discounts, terminated, truncated,
                                               gae_lambda=.8, bootstrap_truncated=True)
    assert torch.allclose(advantage, torch.tensor([1 + .9*.7 - .5 + .9*.8*(2 + .81*3 - .7),
                                                   2 + .81*3 - .7, 100 - .9]))
    assert torch.allclose(returns, advantage + value)
    no_bootstrap, _ = generalized_advantages(reward, value, next_value, discounts, terminated, truncated,
                                            gae_lambda=.8, bootstrap_truncated=False)
    assert no_bootstrap[1] == pytest.approx(1.3)


def test_log_density_modes_have_different_explicit_semantics():
    raw = torch.tensor([[[-10., 8.], [3., 4.]]])
    assert reduce_logprobs(raw, DPPOConfig(), execution_horizon=1).item() == -1.5
    exact = DPPOConfig(logprob_reduction="exact_sum")
    assert reduce_logprobs(raw, exact, execution_horizon=1).item() == 5  # full chain-coordinate density


def test_ppo_clipping_gradient_and_single_denoise_step():
    cfg = DPPOConfig(logprob_reduction="exact_sum", clip_epsilon=.2, clip_epsilon_min=.01)
    assert denoising_clip(torch.tensor([0]), 1, cfg).item() == pytest.approx(.2)
    current = torch.tensor([[[.5]], [[.05]]], requires_grad=True)
    loss, metrics = clipped_policy_loss(current, torch.zeros_like(current), torch.ones(2),
                                       torch.tensor([0, 0]), 1, cfg)
    loss.backward()
    assert current.grad[0].item() == 0
    assert current.grad[1].item() < 0
    assert metrics["clip_fraction"] == .5


def test_value_clipping_preserves_old_prediction_boundary():
    assert clipped_value_loss(torch.tensor([.8]), torch.tensor([0.]), torch.tensor([1.]), .2) == pytest.approx(.32)


def test_bfloat16_master_accumulates_small_updates_and_restores():
    parameter = nn.Parameter(torch.tensor([1.], dtype=torch.bfloat16))
    optimizer = MasterAdamW([parameter], lr=1e-4, weight_decay=0.)
    for _ in range(100):
        optimizer.zero_grad()
        parameter.float().sum().backward()
        optimizer.step(1.)
    assert parameter.item() < .995
    saved = optimizer.state_dict()
    assert saved["masters"][0].dtype == torch.float32
    replacement = nn.Parameter(torch.tensor([8.], dtype=torch.bfloat16))
    resumed = MasterAdamW([replacement], lr=1e-4, weight_decay=0.)
    resumed.load_state_dict(saved)
    assert torch.equal(replacement, parameter)
    assert torch.equal(resumed.masters[0], optimizer.masters[0])


def test_checkpoint_cuda_rng_only_touches_configured_device(monkeypatch):
    from pressb.runtime.dppo_trainer import _device_rng_state, _restore_device_rng
    calls = []
    state = torch.tensor([1, 2, 3], dtype=torch.uint8)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("Must not initialize unrelated GPUs"))
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda *args: pytest.fail("Must not seed unrelated GPUs"))
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: calls.append(("get", str(device))) or state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value, device: calls.append(("set", str(device))))
    assert _device_rng_state("cpu") is None
    assert torch.equal(_device_rng_state("cuda:2"), state)
    _restore_device_rng({"cuda_rng_device": state}, "cuda:2")
    _restore_device_rng({"cuda_rng": [state, state, state]}, "cuda:2")
    assert calls == [("get", "cuda:2"), ("set", "cuda:2"), ("set", "cuda:2")]


class ToyExpert(nn.Module):
    horizon, latent_dim, denoising_steps = 4, 2, 2

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.1))

    @property
    def device(self):
        return self.weight.device

    def describe(self):
        return {"sampler_kind": "test_gaussian", "horizon": self.horizon,
                "latent_dim": self.latent_dim, "denoising_steps": self.denoising_steps}

    def pack_contexts(self, contexts):
        return {"state": torch.stack(contexts)}

    def transition(self, x, steps, condition):
        return x*.1 + self.weight + condition["state"][:, :1, None]*.01, torch.full_like(x, .2)

    def actions_from_latent(self, x):
        return x


class ToyBackend:
    def __init__(self):
        from test_rl_nodes_trainer import metadata
        self.metadata, self.simulation_metadata = metadata(horizon=4, action_dim=2, noise_dim=2, state_dim=2)
        self.expert = ToyExpert()

    def describe(self):
        return deepcopy(self.metadata)

    def encode_batch(self, observations, seeds):
        return [(torch.tensor(row["state"]), np.array([row["state"][0], 2., 3.], np.float32)) for row in observations]

    def decode_batch(self, contexts, noises):
        return [np.zeros((4, 2), np.float32) for _ in contexts]

    def create_action_expert(self, config):
        return self.expert

    def close(self):
        pass


class LocalClient:
    def __init__(self, service):
        self.service, self.calls = service, []
        self.stale = False

    def call(self, route, payload=None):
        self.calls.append((route, deepcopy(payload)))
        result = self.service.handlers[route](payload or {})
        if route == "/sample_chain" and self.stale:
            result["items"][0]["policy_version"] -= 1
        return result


def make_runner(output, *, steps=12, mode="train", checkpoint=None):
    from test_rl_nodes_trainer import Simulation
    backend = ToyBackend()
    service = InferenceService(backend, batch_size=2, action_expert={})
    inference = LocalClient(service)
    simulation = Simulation(backend.simulation_metadata)
    config = dict(model={"factory": "unused:create", "config": {}}, output_dir=str(output), mode=mode,
        rollout=dict(max_transitions=steps, execution_horizon=2, eval_episodes=4),
        dppo=dict(rollout_steps=2, epochs=2, minibatch_size=4, critic_hidden_dim=8, actor_lr=.003,
                  target_kl=None, initial_logprob_atol=.001),
        checkpoint=dict(interval=4, include_replay=False, resume=str(checkpoint) if checkpoint else None))
    trainer = DPPOTrainer(config, inference_client=inference, simulation_client=simulation, expert=ToyExpert())
    return trainer, inference, simulation


def test_real_chain_update_rpc_changes_expert_and_retains_checkpoints(tmp_path):
    runner, client, simulation = make_runner(tmp_path)
    before = state_sha256(runner.expert.state_dict())
    result = runner.run()
    assert result["state"] == "completed" and result["transitions"] == 12
    assert result["updates"] > 0 and result["policy_version"] == 3
    assert state_sha256(runner.expert.state_dict()) != before
    assert state_sha256(runner.expert.state_dict()) == state_sha256(client.service.backend.expert.state_dict())
    assert len(list((tmp_path / "checkpoints").glob("step_*.pt"))) == 3
    payload = torch.load(result["last_checkpoint"], weights_only=True)
    assert "replay" not in payload
    assert payload["actor_optimizer"]["masters"][0].dtype == torch.float32
    sample_versions = [payload["expected_policy_version"] for route, payload in client.calls if route == "/sample_chain"]
    assert sample_versions == [0, 0, 1, 1, 2, 2]
    assert client.service.health()["cache"]["size"] == 0
    assert simulation.closed_run_id == runner.run_id


def test_mixed_or_stale_rollout_rejected_before_physics(tmp_path):
    runner, client, simulation = make_runner(tmp_path)
    client.stale = True
    with pytest.raises(ValueError, match="Mixed policy"):
        runner.run()
    assert not simulation.steps
    assert not [route for route, _ in client.calls if route == "/weights_begin"]


def test_wrong_chunk_discount_rejected(tmp_path):
    runner, _, simulation = make_runner(tmp_path)
    simulation.bad_discount = True
    with pytest.raises(ValueError, match="discount"):
        runner.run()


def test_equal_head_with_different_frozen_backbone_is_rejected(tmp_path, monkeypatch):
    runner, client, simulation = make_runner(tmp_path)
    backend = ToyBackend()
    backend.metadata["source_revision"] = "different-backbone-code"
    assert state_sha256(backend.expert.state_dict()) == state_sha256(runner.expert.state_dict())
    runner.expert = None
    monkeypatch.setattr("pressb.runtime.server.load_backend", lambda *args, **kwargs: backend)
    with pytest.raises(ValueError, match="identity differs: source_revision"):
        runner.run()
    assert not simulation.resets


def test_equal_weights_with_different_expert_implementation_is_rejected(tmp_path):
    runner, client, simulation = make_runner(tmp_path)
    client.service.action_expert.spec["implementation_sha256"]["chain"] = "0" * 64
    with pytest.raises(ValueError, match="specification or implementation differs"):
        runner.run()
    assert not simulation.resets


def test_raw_logprob_check_cannot_be_hidden_by_clipping_or_mean(tmp_path, monkeypatch):
    runner, _, simulation = make_runner(tmp_path)
    original = runner._verify_rollout

    def corrupt_final_pair(records):
        # The last pair is outside the first PPO minibatch. Opposing changes
        # cancel under the mean but must still fail the elementwise check.
        records[-1]["logprobs"][-1, 0, 0] += .5
        records[-1]["logprobs"][-1, 0, 1] -= .5
        return original(records)

    monkeypatch.setattr(runner, "_verify_rollout", corrupt_final_pair)
    with pytest.raises(ValueError, match="log-probability mismatch"):
        runner.run()
    assert runner.updates == 0


def test_raw_check_rejects_densities_below_surrogate_clamp(tmp_path, monkeypatch):
    runner, _, _ = make_runner(tmp_path)
    runner.ft_steps = runner.total_steps = 2
    runner.contract = {"execution_horizon": 2}
    old = torch.full((2, 4, 2), -10.)
    current = torch.full_like(old, -100.)
    assert torch.equal(reduce_logprobs(old, runner.config.dppo), reduce_logprobs(current, runner.config.dppo))
    monkeypatch.setattr(runner, "_pair_batch", lambda *args: (None, None, None, old, current))
    with pytest.raises(ValueError, match="raw_max=90"):
        runner._verify_rollout([{}])
    assert runner.updates == 0


def test_resume_does_not_reuse_partially_completed_episode_seed(tmp_path):
    runner, _, _ = make_runner(tmp_path / "first", steps=4)
    path = runner.run()["last_checkpoint"]
    payload = torch.load(path, weights_only=True)
    assert payload["resume_episode_serial"][1] > payload["episode_serial"][1]
    resumed, _, sim = make_runner(tmp_path / "resumed", steps=8, checkpoint=path)
    resumed.run()
    env1 = [row for row in sim.resets if row["env_ids"] == [1]][0]
    assert env1["seed"] == resumed._seed(1, payload["resume_episode_serial"][1], "reset")


def test_physics_failure_is_never_retried(tmp_path):
    runner, _, simulation = make_runner(tmp_path)
    simulation.lose_step_response = True
    with pytest.raises(TimeoutError):
        runner.run()
    assert len(simulation.steps) == 1
    assert runner.health()["execution_state"].startswith("unknown")


def test_checkpoint_resume_publishes_before_sampling(tmp_path):
    first, _, _ = make_runner(tmp_path / "first", steps=4)
    checkpoint = first.run()["last_checkpoint"]
    runner, client, _ = make_runner(tmp_path / "second", steps=8, checkpoint=checkpoint)
    result = runner.run()
    assert result["transitions"] == 8
    routes = [route for route, _ in client.calls]
    assert routes.index("/weights_commit") < routes.index("/sample_chain")
    assert result["updates"] > first.updates


def test_deterministic_eval_has_per_env_quotas_and_stable_seeds(tmp_path):
    first, client1, simulation1 = make_runner(tmp_path / "first", mode="eval", steps=40)
    result = first.run()
    second, client2, simulation2 = make_runner(tmp_path / "second", mode="eval", steps=40)
    result2 = second.run()
    assert result["episodes"] == result2["episodes"] == 4
    assert first.episode_serial == {0: 2, 1: 2}
    assert [row["seed"] for row in simulation1.resets] == [row["seed"] for row in simulation2.resets]
    samples1 = [payload for route, payload in client1.calls if route == "/sample_chain"]
    samples2 = [payload for route, payload in client2.calls if route == "/sample_chain"]
    assert [item["seeds"] for item in samples1] == [item["seeds"] for item in samples2]
    assert all(item["deterministic"] for item in samples1)
    assert not any(route.startswith("/weights_") for route, _ in client1.calls)
    negotiated = json.loads((tmp_path / "first/negotiated.json").read_text())
    assert negotiated["rollout_seed_schedule"]["algorithm"] == "blake2b"
    resets = [json.loads(line) for line in (tmp_path / "first/resets.jsonl").read_text().splitlines()]
    for row in resets:
        assert row["policy_seed_at_step0"] == first._seed(row["env_id"], row["episode_index"], "policy", 0)


@pytest.mark.parametrize("change", [{"rollout": {"fused_step_reset": True}},
                                    {"checkpoint": {"include_replay": True}},
                                    {"checkpoint": {"keep": 2}}, {"dppo": {"actor_lr": -1}},
                                    {"dppo": {"logprob_reduction": "unknown"}}])
def test_invalid_config_is_rejected(change):
    with pytest.raises(ValueError):
        DPPOTrainerConfig(model={"plugin": "diffusion_policy"}, **change)


def test_experiment_compiler_injects_model_and_expert(tmp_path):
    from pressb.experiments.config import load_experiment
    from pressb.experiments.cli import main
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps({"schema_version": 1, "name": "dppo_test", "task": {"plugin": "pap"},
        "model": {"plugin": "walloss", "config": {"checkpoint": "same.safetensors"}},
        "run": {"method": "dppo", "action_expert": {"transition_std": .1}}}))
    experiment = load_experiment(path)
    assert experiment["nodes"]["trainer"]["model"] == experiment["nodes"]["inference"]["backend"]
    assert experiment["nodes"]["inference"]["service"]["action_expert"] == {"transition_std": .1}
    assert main(["validate", "--config", str(path)]) == 0
