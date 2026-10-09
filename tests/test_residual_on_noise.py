"""CPU contracts for fresh residual learning on an immutable learned-noise base."""
from copy import deepcopy
from dataclasses import asdict
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pressb.online_rl.fast_runner import FastOnlineRunner, FastRunConfig
from pressb.online_rl.learner import (CHECKPOINT_VERSION, SOURCE_COMMIT,
                                    LearnerConfig, SACLearner, _Actor)
from pressb.online_rl.protocol import pose8_to_pose9, pose9_to_pose8
from pressb.online_rl.residual_on_noise import (
    BudgetReplayBuffer, FrozenNoisePolicy, ResidualOnNoiseRunner, actor_sha256,
    file_sha256,
)
from test_fast_online_rl_runner import BatchInference, IndexedSimulation, config
from test_online_rl_runner import TOKEN, serve


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def noise_checkpoint(path, *, obs_dim=4, identities=None, deterministic=False):
    cfg = LearnerConfig(method="initial_noise", obs_dim=obs_dim, hidden_dim=8)
    actor = _Actor(obs_dim, cfg.action_dim, cfg)
    payload = dict(version=CHECKPOINT_VERSION, source_commit=SOURCE_COMMIT,
                   config=asdict(cfg), actor=actor.state_dict(), updates=71,
                   extra=dict(identities=identities or {"model": "fixture"},
                              run_config={"deterministic_eval": deterministic}),
                   # Deliberately unusable training state: the frozen loader
                   # must neither construct critics nor restore optimizer/RNG.
                   qs="not a critic", actor_optimizer="not an optimizer",
                   rng="must not be restored")
    torch.save(payload, path)
    return payload, actor_sha256(actor)


def test_frozen_actor_load_and_seeded_sampling_do_not_pollute_rng(tmp_path, monkeypatch):
    path = tmp_path / "noise.pt"
    _, expected_hash = noise_checkpoint(path)
    monkeypatch.setattr(SACLearner, "load", lambda *a, **k:
                        pytest.fail("Frozen policy must not restore a learner"))
    before_torch = torch.get_rng_state().clone()
    before_numpy, before_python = deepcopy(np.random.get_state()), random.getstate()
    policy = FrozenNoisePolicy(path, device="cpu", identities={"model": "fixture"}, obs_dim=4)
    torch.testing.assert_close(torch.get_rng_state(), before_torch, rtol=0, atol=0)
    vectors = np.arange(12, dtype=np.float32).reshape(3, 4) / 12
    seeds = [101, 202, 303]
    first = policy.act(vectors, seeds)
    np.testing.assert_array_equal(first, policy.act(vectors, seeds))
    # Sampling belongs to an observation's seed, independent of batch order.
    np.testing.assert_array_equal(first[[2, 0]], policy.act(vectors[[2, 0]], [303, 101]))
    assert not np.array_equal(first, policy.act(vectors, [102, 203, 304]))
    assert first.shape == (3, 9) and np.abs(first).max() <= 1
    torch.testing.assert_close(torch.get_rng_state(), before_torch, rtol=0, atol=0)
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0] and before_numpy[2:] == after_numpy[2:]
    np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
    assert random.getstate() == before_python
    assert not policy.actor.training
    assert all(not p.requires_grad and p.grad is None for p in policy.actor.parameters())
    assert policy.verify_frozen() == expected_hash
    assert policy.source_updates == 71 and policy.checkpoint_sha256 == file_sha256(path)


def noise_identities(gamma=.99):
    return dict(
        runner_contract="independent_episodes_pipeline_v1",
        inference=dict(checkpoint_sha256="model-weights", source_sha256={"model.py": "model-code"}),
        simulation=dict(single_gamma=gamma, physics_dt=1 / 120, smoothing_window=3,
                        source_sha256={"simulation.py": "simulation-code"},
                        snapshot_sha256="fixed-scene"),
    )


def test_frozen_noise_can_transfer_gamma_without_changing_actor_identity_or_rng(tmp_path):
    path = tmp_path / "noise.pt"
    source = noise_identities(.99)
    runtime = noise_identities(.995)
    _, expected_hash = noise_checkpoint(path, identities=source)
    before_torch = torch.get_rng_state().clone()
    before_numpy, before_python = deepcopy(np.random.get_state()), random.getstate()
    original = FrozenNoisePolicy(path, device="cpu", identities=source, obs_dim=4)
    transferred = FrozenNoisePolicy(path, device="cpu", identities=runtime, obs_dim=4)
    vectors = np.arange(12, dtype=np.float32).reshape(3, 4) / 12
    np.testing.assert_array_equal(original.act(vectors, [1, 2, 3]),
                                  transferred.act(vectors, [1, 2, 3]))
    assert transferred.verify_frozen() == expected_hash
    assert transferred.identity == original.identity
    # The established checkpoint identity must not gain fields: otherwise old
    # residual-on-noise checkpoints could no longer be evaluated at gamma=.99.
    assert set(transferred.identity) == {
        "contract", "checkpoint_sha256", "actor_sha256", "noise_steps", "noise_scale",
        "deterministic", "seed_contract", "base_action_contract",
    }
    assert transferred.objective_transfer == dict(
        source_single_gamma=.99, runtime_single_gamma=.995, changed=True)
    assert original.objective_transfer == dict(
        source_single_gamma=.99, runtime_single_gamma=.99, changed=False)
    assert source == noise_identities(.99) and runtime == noise_identities(.995)
    torch.testing.assert_close(torch.get_rng_state(), before_torch, rtol=0, atol=0)
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0] and before_numpy[2:] == after_numpy[2:]
    np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
    assert random.getstate() == before_python


@pytest.mark.parametrize("field", [
    ("runner_contract",), ("inference", "checkpoint_sha256"),
    ("inference", "source_sha256", "model.py"), ("simulation", "physics_dt"),
    ("simulation", "smoothing_window"), ("simulation", "snapshot_sha256"),
    ("simulation", "source_sha256", "simulation.py"),
])
def test_noise_gamma_transfer_keeps_all_other_identity_fields_strict(tmp_path, field):
    path = tmp_path / "noise.pt"
    noise_checkpoint(path, identities=noise_identities(.99))
    runtime = noise_identities(.995)
    target = runtime
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = "different"
    with pytest.raises(ValueError, match="different model or simulator contract"):
        FrozenNoisePolicy(path, device="cpu", identities=runtime, obs_dim=4)


@pytest.mark.parametrize("side", ["source", "runtime"])
@pytest.mark.parametrize("gamma", [0., -.01, 1.001, float("nan"), float("inf"), True, ".995", None])
def test_noise_gamma_transfer_rejects_invalid_discount(tmp_path, side, gamma):
    path = tmp_path / "noise.pt"
    source, runtime = noise_identities(.99), noise_identities(.995)
    (source if side == "source" else runtime)["simulation"]["single_gamma"] = gamma
    noise_checkpoint(path, identities=source)
    with pytest.raises(ValueError):
        FrozenNoisePolicy(path, device="cpu", identities=runtime, obs_dim=4)


@pytest.mark.parametrize("damage", ["weights", "requires_grad", "gradient"])
def test_freeze_verification_detects_mutation_or_gradient(tmp_path, damage):
    path = tmp_path / "noise.pt"
    noise_checkpoint(path)
    policy = FrozenNoisePolicy(path, device="cpu", identities={"model": "fixture"}, obs_dim=4)
    parameter = next(policy.actor.parameters())
    if damage == "weights":
        with torch.no_grad():
            parameter.add_(1)
    elif damage == "requires_grad":
        parameter.requires_grad_(True)
    else:
        parameter.grad = torch.zeros_like(parameter)
    with pytest.raises(RuntimeError, match="modified or received gradients"):
        policy.verify_frozen()


@pytest.mark.parametrize("change", ["version", "source", "method", "dimension", "identities", "sampling"])
def test_incompatible_frozen_noise_checkpoint_fails_closed(tmp_path, change):
    path = tmp_path / "noise.pt"
    payload, _ = noise_checkpoint(path)
    if change == "version":
        payload["version"] = -1
    elif change == "source":
        payload["source_commit"] = "different"
    elif change == "method":
        payload["config"]["method"] = "action_residual"
    elif change == "dimension":
        payload["config"]["obs_dim"] = 5
    elif change == "identities":
        payload["extra"]["identities"] = {"model": "other"}
    else:
        payload["extra"]["run_config"]["deterministic_eval"] = "false"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        FrozenNoisePolicy(path, device="cpu", identities={"model": "fixture"}, obs_dim=4)


@pytest.mark.parametrize("override", [dict(resume=True), dict(warmstart_actor="old.pt"),
                                      dict(checkpoint="old.pt")])
def test_training_rejects_residual_restore_before_allocating_runner(monkeypatch, override):
    monkeypatch.setattr(FastOnlineRunner, "__init__", lambda *a, **k:
                        pytest.fail("Reject restore before opening output or RPC resources"))
    with pytest.raises(ValueError, match="from scratch|Fresh residual"):
        ResidualOnNoiseRunner(FastRunConfig(), noise_checkpoint="noise.pt", **override)


class NoiseSpy:
    config = SimpleNamespace(noise_steps=1, noise_scale=1.5)

    def __init__(self):
        self.calls = []

    def act(self, vectors, seeds):
        self.calls.append((vectors.copy(), list(seeds)))
        return np.full((len(vectors), 9), .125, dtype=np.float32)


class ResidualSpy:
    config = SimpleNamespace(residual_scale=(.03, .03, .03, .1, .1, .1, .1, .1, .1))

    def __init__(self):
        self.calls = []

    def act(self, vectors, base, **kwargs):
        self.calls.append((vectors.copy(), base.copy(), kwargs))
        actions = np.zeros((len(vectors), 7, 9), dtype=np.float32)
        if not kwargs["warmup"]:
            actions[:, :, 0], actions[:, :, 2] = .5, -.25
        return actions.reshape(len(vectors), 63)


class DecoderSpy:
    def __init__(self):
        self.requests, self.responses = [], {}
        self.counter = 0
        self.mismatch = False

    def call(self, path, payload):
        self.requests.append((path, deepcopy(payload)))
        if path == "/encode":
            assert payload["include_base_actions"] is False
            rows = []
            for obs in payload["observations"]:
                self.counter += 1
                rows.append(dict(context_id=f"ctx-{self.counter}",
                                 feature=np.zeros(2048).tolist(),
                                 state_pose9=pose8_to_pose9(obs["state"]).tolist()))
            return {"items": rows}
        assert path == "/decode"
        rows = []
        for item in payload["items"]:
            np.testing.assert_array_equal(item["initial_noise"], np.full((7, 9), .1875))
            serial = int(item["context_id"].split("-")[-1])
            # Non-orthonormal raw rotation6D and rounded wire quaternions catch
            # inadvertent pose8 roundtrips and zero-residual re-projection.
            raw = np.tile([.25 + serial * .01, -.04, .18, 2., .3, 0., .4, 1.5, .1], (7, 1))
            wire = pose9_to_pose8(raw).astype(np.float32).tolist()
            row = dict(context_id=item["context_id"], actions_pose9=raw.tolist(), actions_pose8=wire)
            self.responses[item["context_id"]] = deepcopy(row)
            rows.append(row)
        return {"items": list(reversed(rows)) if self.mismatch else rows}


@pytest.fixture
def prepared_runner(tmp_path):
    runner = ResidualOnNoiseRunner(
        FastRunConfig(pipeline_updates=False, learning_start=2, progressive_exploration=0),
        "http://unused", "http://unused", tmp_path / "run", noise_checkpoint=tmp_path / "noise.pt")
    runner.obs_dim, runner.feature_encoding, runner.fused_release = 2106, "json", False
    decoder = DecoderSpy()
    runner.inference.client = decoder
    runner.noise_policy, runner.learner = NoiseSpy(), ResidualSpy()
    items = [dict(env_id=eid, episode_id=f"episode-{eid}",
                  observation=dict(task=f"Press {24 + eid} floor.",
                                   state=[.2, 0., .18, 1., 0., 0., 0., .008],
                                   control_state=np.zeros(37).tolist(), images={})) for eid in (9, 2)]
    runner.episode_seed_indices = {f"episode-{eid}": eid for eid in (9, 2)}
    try:
        yield runner, decoder, items
    finally:
        runner.executor.shutdown(wait=True, cancel_futures=True)


def test_encode_preserves_raw_noise_actions_and_warmup_executes_wire_exactly(prepared_runner):
    runner, decoder, items = prepared_runner
    prepared = runner._encode(items)
    assert [name for name, _ in decoder.requests] == ["/encode", "/decode"]
    assert len(runner.noise_policy.calls) == 1
    assert runner.noise_policy.calls[0][1] == [42 + 2 * 10000, 42 + 9 * 10000]
    actions, normalized = runner._actions(prepared)
    assert [row["env_id"] for row in actions] == [2, 9]
    for row in actions:
        encoded = prepared[row["env_id"]][1]
        decoded = decoder.responses[encoded["context_id"]]
        np.testing.assert_array_equal(encoded["base_actions_pose9"], decoded["actions_pose9"])
        assert not np.allclose(encoded["base_actions_pose9"], pose8_to_pose9(decoded["actions_pose8"]))
        np.testing.assert_array_equal(row["actions_pose8"], decoded["actions_pose8"])
        assert not np.any(normalized[row["env_id"]])
    assert runner.learner.calls[0][2]["warmup"] is True
    assert len(decoder.requests) == 2 and len(runner.noise_policy.calls) == 1


def test_cached_next_base_is_next_actual_residual_condition_without_resampling(prepared_runner):
    runner, decoder, items = prepared_runner
    previous = runner._encode(items)
    current = runner._encode(deepcopy(items))
    replay = BudgetReplayBuffer(4, 2106, 63, max_transitions=4)
    for eid in sorted(previous):
        replay.add(previous[eid][2], np.zeros(63), 0., .99, current[eid][2], False, False,
                   base_action=previous[eid][1]["base_actions_pose9"],
                   next_base_action=current[eid][1]["base_actions_pose9"])
    runner.status["transitions"] = 2
    actions, _ = runner._actions(current)
    base = runner.learner.calls[-1][1]
    np.testing.assert_array_equal(base.reshape(2, 63).astype(np.float32),
                                  replay.arrays["next_base_action"][:2])
    residual = np.zeros((2, 7, 9))
    residual[:, :, 0], residual[:, :, 2] = .5, -.25
    np.testing.assert_allclose([row["actions_pose8"] for row in actions],
        pose9_to_pose8(base + residual * runner.learner.config.residual_scale), atol=1e-12)
    assert [name for name, _ in decoder.requests] == ["/encode", "/decode", "/encode", "/decode"]
    assert len(runner.noise_policy.calls) == 2
    assert runner.noise_policy.calls[1][1] == [42 + 2 * 10000 + 1, 42 + 9 * 10000 + 1]


def test_noise_decode_context_mismatch_rejected(prepared_runner):
    runner, decoder, items = prepared_runner
    decoder.mismatch = True
    with pytest.raises(ValueError, match="mismatched contexts"):
        runner._encode(items)
    assert not runner.learner.calls


def test_replay_budget_survives_crossing_batches_and_ring_wrap():
    replay = BudgetReplayBuffer(3, 4, 63, max_transitions=5)
    base = np.zeros((7, 9))
    for batch in ([0, 1], [2, 3], [4, 5], [6, 7]):
        for marker in batch:
            replay.add(np.full(4, marker), np.zeros(63), 0., .9, np.full(4, marker + 1),
                       False, False, base_action=base, next_base_action=base)
    assert replay.accepted_transitions == 5 and len(replay) == 3 and replay.position == 2
    np.testing.assert_array_equal(replay.arrays["obs"][:, 0], [3, 4, 2])


def test_400k_budget_clamps_crossing_batch_credit_to_398k(monkeypatch):
    runner = ResidualOnNoiseRunner.__new__(ResidualOnNoiseRunner)
    runner.config = FastRunConfig(max_transitions=400000, learning_start=2000,
                                  training_frequency=1, utd=1., pipeline_updates=True)
    runner.update_credit = runner.earned_update_credit = 0.
    runner.update_budget = runner.last_training_bucket = 0
    runner.training_budget_reached = False
    runner.learner = SimpleNamespace(updates=0, actor=object())
    runner.noise_policy = SimpleNamespace(verify_frozen=lambda: "frozen-hash")
    events, flushes = [], []
    runner._append = lambda name, row: events.append(row)
    monkeypatch.setattr("pressb.online_rl.residual_on_noise.actor_sha256", lambda _: "residual-hash")

    def flush():
        flushes.append(runner.update_credit)
        runner.learner.updates += int(runner.update_credit)
        runner.update_credit %= 1
        runner.update_budget = 0

    runner._flush_updates = flush
    for before, after in [(0, 1984), (1984, 2048), (2048, 399984),
                          (399984, 400048), (400048, 400112)]:
        runner._accrue_updates(before, after)
    assert runner.earned_update_credit == runner.learner.updates == 398000
    assert runner.update_credit == runner.update_budget == 0
    assert flushes == [398000] and runner.training_budget_reached
    assert len(events) == 1 and events[0]["training_transitions"] == 400000


class GammaIndexedSimulation(IndexedSimulation):
    """Expose and apply the same reward/discount objective as the real service."""

    def __init__(self, single_gamma=.99):
        super().__init__()
        self.single_gamma = single_gamma

    def health(self, payload):
        return {**super().health(payload), "single_gamma": self.single_gamma}

    def step(self, payload):
        result = super().step(payload)
        for row in self.items:
            ticks = row["executed_physics_steps"]
            row["discount"] = self.single_gamma ** (ticks / 4)
            row["reward"] = (self.single_gamma ** ((ticks - 1) / 4)
                             if ticks and row["info"]["success"] else 0.)
        result["items"] = deepcopy(self.items)
        return result


class LongerIndexedSimulation(GammaIndexedSimulation):
    """Keep env 1 live for five chunks, including three post-budget RPCs."""

    def step(self, payload):
        result = super().step(payload)
        if any(row["env_id"] == 1 for row in payload["actions"]):
            timeout = self.ages[1] == 5
            ticks = 5 if timeout else 28
            self.items[1].update(truncated=timeout, executed_physics_steps=ticks,
                                 discount=self.single_gamma ** (ticks / 4))
            result["items"] = deepcopy(self.items)
            result["all_done"] = all(row["terminated"] or row["truncated"] for row in self.items)
        return result


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("budget,total", [(5, 6), (3, 7)])
@pytest.mark.parametrize("gamma", [.99, .995])
def test_real_cpu_updates_change_only_fresh_residual_and_honor_budget(tmp_path, pipeline, budget, total, gamma):
    simulation = LongerIndexedSimulation(gamma) if budget == 3 else GammaIndexedSimulation(gamma)
    inference = BatchInference()
    path = tmp_path / "noise.pt"
    with serve(simulation.handlers) as sim_url, serve(inference.handlers) as infer_url:
        runner = ResidualOnNoiseRunner(
            config(max_transitions=budget, learning_start=2, pipeline_updates=pipeline, single_gamma=gamma),
            sim_url, infer_url, tmp_path / "run", noise_checkpoint=path,
            device="cpu", token=TOKEN, timeout=5)
        try:
            runner._verify_nodes()
            noise_source = deepcopy(runner.identities)
            noise_source["simulation"]["single_gamma"] = .99
            _, frozen_hash = noise_checkpoint(path, obs_dim=runner.obs_dim, identities=noise_source)
            checkpoint_hash = file_sha256(path)
            result = runner.run()
        finally:
            runner.executor.shutdown(wait=True, cancel_futures=True)
    assert result["state"] == "complete"
    assert result["transitions"] == total and result["training_transitions"] == budget
    assert result["drain_transitions"] == total - budget and result["updates"] == budget - 2
    assert result["earned_update_credit"] == budget - 2 and result["update_credit"] == 0
    assert runner.replay.accepted_transitions == len(runner.replay) == budget
    assert runner.noise_policy.verify_frozen() == frozen_hash and file_sha256(path) == checkpoint_hash
    initial = torch.load(runner.output / "initial.pt", map_location="cpu", weights_only=False)
    assert initial["updates"] == 0 and initial["extra"]["replay_size"] == 0
    assert all(not initial[name]["state"] for name in
               ("actor_optimizer", "q_optimizer", "alpha_optimizer"))
    assert any(not torch.equal(initial["actor"][name], value)
               for name, value in runner.learner.actor.state_dict().items())
    assert all(parameter.grad is None for parameter in runner.noise_policy.actor.parameters())
    assert all("base_actions_pose9" not in row["item"] for row in inference.encoded)
    assert len(inference.decoded) == len(inference.encoded)
    assert not inference.contexts and not runner.contexts
    arrays = runner.replay.state_dict()["arrays"]
    ticks = np.array([3, 28, 3, 28, 3])[:budget]
    np.testing.assert_allclose(arrays["discount"][:, 0], gamma ** (ticks / 4), rtol=1e-7)
    np.testing.assert_allclose(arrays["reward"][:, 0],
                              np.where(ticks == 3, gamma ** ((ticks - 1) / 4), 0.), rtol=1e-7)
    composition = json.loads((runner.output / "composition.json").read_text())
    assert composition["frozen_noise_objective_transfer"] == dict(
        source_single_gamma=.99, runtime_single_gamma=gamma, changed=gamma != .99)
    assert initial["extra"]["identities"]["simulation"]["single_gamma"] == gamma
    # Env 1 continues across a reset of env 0. Its cached terminal-free
    # next action must be identical to the following transition's base.
    if budget > 3:
        np.testing.assert_array_equal(arrays["next_base_action"][1], arrays["base_action"][3])
    np.testing.assert_array_equal(arrays["action"][:2], 0.)
    for eid, row in enumerate(simulation.steps[0]["actions"]):
        expected = pose9_to_pose8(arrays["base_action"][eid].reshape(7, 9))
        np.testing.assert_allclose(row["actions_pose8"], expected, atol=1e-7)
    verification = json.loads((runner.output / "freeze_verification.json").read_text())
    assert verification["noise_checkpoint_unchanged"] and verification["residual_updates"] == budget - 2
    events = [json.loads(line) for line in (runner.output / "events.jsonl").read_text().splitlines()]
    boundary = [row for row in events if row["event"] == "training_budget_reached"]
    assert len(boundary) == 1 and boundary[0]["updates"] == budget - 2
    assert boundary[0]["residual_actor_sha256"] == actor_sha256(runner.learner.actor)
    if budget == 3:
        # Three real further simulation RPCs occur after the budget boundary;
        # these transitions are drained with unchanged residual weights.
        assert [len(step["actions"]) for step in simulation.steps] == [2, 2, 1, 1, 1]


@pytest.mark.parametrize("basis", ["same", "changed_noise", "legacy_residual", "changed_gamma"])
def test_trained_checkpoint_evaluation_requires_same_noise_basis_and_never_updates(tmp_path, basis):
    noise_path = tmp_path / "noise.pt"
    simulation, inference = IndexedSimulation(), BatchInference()
    with serve(simulation.handlers) as sim_url, serve(inference.handlers) as infer_url:
        trained = ResidualOnNoiseRunner(
            config(max_transitions=3, learning_start=2, pipeline_updates=False),
            sim_url, infer_url, tmp_path / "train", noise_checkpoint=noise_path,
            device="cpu", token=TOKEN, timeout=5)
        try:
            trained._verify_nodes()
            base_identities = deepcopy(trained.identities)
            noise_checkpoint(noise_path, obs_dim=trained.obs_dim, identities=base_identities)
            trained.run()
        finally:
            trained.executor.shutdown(wait=True, cancel_futures=True)
    residual_path = trained.output / "last.pt"
    expected_actor_hash = actor_sha256(trained.learner.actor)
    expected_noise_hash = trained.noise_policy.verify_frozen()
    expected_updates = trained.learner.updates
    if basis == "changed_noise":
        changed = torch.load(noise_path, map_location="cpu", weights_only=False)
        next(iter(changed["actor"].values())).add_(.01)
        noise_path = tmp_path / "other-noise.pt"
        torch.save(changed, noise_path)
    elif basis == "legacy_residual":
        legacy = torch.load(residual_path, map_location="cpu", weights_only=False)
        legacy["extra"]["identities"].pop("frozen_initial_noise")
        residual_path = tmp_path / "original-base-residual.pt"
        torch.save(legacy, residual_path)

    evaluation_gamma = .995 if basis == "changed_gamma" else .99
    simulation, inference = GammaIndexedSimulation(evaluation_gamma), BatchInference()
    with serve(simulation.handlers) as sim_url, serve(inference.handlers) as infer_url:
        evaluated = ResidualOnNoiseRunner(
            config(mode="eval", eval_episodes=2, pipeline_updates=False, single_gamma=evaluation_gamma),
            sim_url, infer_url, tmp_path / "eval", noise_checkpoint=noise_path,
            checkpoint=residual_path, device="cpu", token=TOKEN, timeout=5)
        try:
            if basis != "same":
                with pytest.raises(ValueError, match="different frozen model or simulator contract"):
                    evaluated.run()
                assert not simulation.resets and not inference.encoded
                return
            result = evaluated.run()
        finally:
            evaluated.executor.shutdown(wait=True, cancel_futures=True)
    assert result["state"] == "complete" and result["episodes"] == 2
    assert evaluated.learner.updates == expected_updates
    assert len(evaluated.replay) == evaluated.earned_update_credit == 0
    assert actor_sha256(evaluated.learner.actor) == expected_actor_hash
    assert evaluated.noise_policy.verify_frozen() == expected_noise_hash
    assert all(not parameter.requires_grad for module in
               (evaluated.learner.actor, evaluated.learner.qs, evaluated.learner.q_targets)
               for parameter in module.parameters())
    assert not evaluated.learner.log_alpha.requires_grad
    assert not inference.contexts and not evaluated.contexts
