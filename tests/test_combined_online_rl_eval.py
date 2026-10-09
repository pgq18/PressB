"""CPU policy-composition checks; simulator and frozen VLA RPC are synthetic."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pressb.online_rl.fast_runner import FastOnlineRunner, FastRunConfig
from pressb.online_rl.protocol import pose8_to_pose9, pose9_to_pose8


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/eval_combined_online_rl.py"
SPEC = importlib.util.spec_from_file_location("combined_eval_under_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CombinedEvaluationRunner = MODULE.CombinedEvaluationRunner


class NoiseActor:
    device = torch.device("cpu")
    config = SimpleNamespace(noise_steps=1, noise_scale=1.5)

    def __init__(self):
        self.calls = []

    def act(self, obs, *, deterministic):
        self.calls.append((obs.copy(), deterministic))
        return np.full((len(obs), 9), .125, np.float32)


class ResidualActor:
    config = SimpleNamespace(residual_scale=(.03, .03, .03, .1, .1, .1, .1, .1, .1))

    def __init__(self, actions):
        self.actions = actions
        self.calls = []

    def act(self, obs, base_actions, *, deterministic):
        self.calls.append((obs.copy(), base_actions.copy(), deterministic))
        return self.actions.copy()


def composition_fixture(monkeypatch, *, zero_residual=False, mismatched_context=False):
    runner = CombinedEvaluationRunner.__new__(CombinedEvaluationRunner)
    runner.config = FastRunConfig(method="initial_noise", mode="eval", pipeline_updates=False,
                                  progressive_exploration=0, deterministic_eval=False)
    runner.status = {"transitions": 0}
    runner.learner = NoiseActor()
    # Non-unit, non-orthogonal rotation6D rows ensure that a pose8 roundtrip
    # would destroy information which the trained residual actor conditions on.
    raw = np.tile([.25, -.04, .18, 2., .3, 0., .4, 1.5, .1], (2, 7, 1))
    raw[1, :, :3] += [.07, .02, -.015]
    raw[:, :, 0] += np.arange(7)[None] * .001
    decoded = []
    for index, eid in enumerate([2, 9]):
        # Deliberate float32 wire rounding must survive a zero residual exactly.
        wire = pose9_to_pose8(raw[index]).astype(np.float32).tolist()
        decoded.append(dict(context_id=f"ctx-{eid}", actions_pose9=raw[index].tolist(),
                            actions_pose8=wire))
    if mismatched_context:
        decoded = list(reversed(decoded))
    residual = np.zeros((2, 7, 9), np.float32)
    if not zero_residual:
        residual[:, :, 0] = .5
        residual[:, :, 2] = -.25
        residual[:, :, 4] = .2
    runner.residual_learner = ResidualActor(residual.reshape(2, 63))
    records, rpc_requests = [], []
    runner._append = lambda name, row: records.append((name, row))
    runner.episode_seed_indices = {f"episode-{eid}": index for index, eid in enumerate([2, 9])}

    def parent_rpc(self, client, service, path, payload):
        assert self is runner and service == "inference" and path == "/decode"
        rpc_requests.append(deepcopy(payload))
        return {"items": deepcopy(decoded)}

    # Exercise CombinedEvaluationRunner._call capture and the actual inherited
    # noise _actions implementation, including its context-order validation.
    monkeypatch.setattr(FastOnlineRunner, "_call", parent_rpc)
    runner.inference = SimpleNamespace(call=lambda path, payload:
                                      runner._call(None, "inference", path, payload))
    active = {}
    for eid in [9, 2]:  # Deliberately unsorted environment insertion order.
        active[eid] = (dict(episode_id=f"episode-{eid}", info={"physics_index": 28}),
                       dict(context_id=f"ctx-{eid}", rollout_seed=20260930 + eid * 10000),
                       np.full(2106, eid, dtype=np.float32))
    return runner, active, raw, residual, decoded, records, rpc_requests


def test_noise_raw_pose9_conditions_residual_and_is_addition_origin(monkeypatch):
    runner, active, raw, residual, decoded, records, requests = composition_fixture(monkeypatch)
    rows, noise = runner._actions(active)

    assert [row["env_id"] for row in rows] == [2, 9]
    obs, conditioned_base, deterministic = runner.residual_learner.calls[0]
    np.testing.assert_array_equal(obs[:, 0], [2, 9])
    np.testing.assert_array_equal(conditioned_base, raw)
    assert deterministic is True
    roundtripped = np.stack([pose8_to_pose9(row["actions_pose8"]) for row in decoded])
    assert not np.allclose(conditioned_base[:, :, 3:], roundtripped[:, :, 3:])
    expected = raw + residual * np.asarray(runner.residual_learner.config.residual_scale)
    np.testing.assert_allclose(np.asarray([row["actions_pose8"] for row in rows]),
                               pose9_to_pose8(expected), rtol=0, atol=1e-12)
    np.testing.assert_allclose(expected[:, :, :3] - raw[:, :, :3],
                               np.broadcast_to([.015, 0., -.0075], (2, 7, 3)), atol=1e-12)
    assert all(call[1] is False for call in runner.learner.calls)
    assert set(noise) == {2, 9}
    assert len(requests) == 1
    assert [item["context_id"] for item in requests[0]["items"]] == ["ctx-2", "ctx-9"]
    np.testing.assert_array_equal(np.asarray(requests[0]["items"][0]["initial_noise"]),
                                  np.full((7, 9), .1875))
    for index, (name, record) in enumerate(records):
        assert name == "policy_actions.jsonl"
        np.testing.assert_array_equal(record["noise_actions_pose9"], raw[index])
        np.testing.assert_allclose(record["combined_actions_pose9"], expected[index], atol=1e-12)


def test_zero_residual_preserves_noise_wire_pose8_exactly(monkeypatch):
    runner, active, raw, _, decoded, _, _ = composition_fixture(monkeypatch, zero_residual=True)
    rows, _ = runner._actions(active)
    for index, row in enumerate(rows):
        np.testing.assert_array_equal(row["actions_pose8"], decoded[index]["actions_pose8"])
    assert not np.array_equal(np.asarray(rows[0]["actions_pose8"]), pose9_to_pose8(raw[0]))


def test_mismatched_decode_contexts_rejected_before_residual_actor(monkeypatch):
    runner, active, _, _, _, records, _ = composition_fixture(monkeypatch, mismatched_context=True)
    with pytest.raises(ValueError, match="mismatched contexts"):
        runner._actions(active)
    assert not runner.residual_learner.calls and not records


def test_stale_captured_context_rejected_even_if_parent_returns_actions(monkeypatch):
    runner, active, _, _, decoded, records, _ = composition_fixture(monkeypatch)
    runner._last_noise_decode = [{**decoded[0], "context_id": "stale"}, decoded[1]]
    monkeypatch.setattr(FastOnlineRunner, "_actions", lambda self, items:
                        ([dict(env_id=eid, actions_pose8=decoded[index]["actions_pose8"])
                          for index, eid in enumerate([2, 9])],
                         {eid: np.zeros(9) for eid in [2, 9]}))
    with pytest.raises(ValueError, match="Captured noise decoder contexts"):
        runner._actions(active)
    assert not runner.residual_learner.calls and not records


@pytest.mark.parametrize("mode,method", [("train", "initial_noise"),
                                         ("eval", "action_residual"), ("eval", "base")])
def test_composition_rejects_training_and_non_noise_configuration(monkeypatch, mode, method):
    # No executor or HTTP client is needed to exercise the composition guard.
    monkeypatch.setattr(FastOnlineRunner, "__init__", lambda self, config, *args, **kwargs:
                        setattr(self, "config", config))
    config = FastRunConfig(method=method, mode=mode)
    with pytest.raises(ValueError, match="evaluation-only"):
        CombinedEvaluationRunner(config, residual_checkpoint="unused-residual.pt")


def test_evaluation_neither_earns_updates_nor_allows_direct_update(monkeypatch):
    runner, *_ = composition_fixture(monkeypatch)
    runner.update_credit = 0.
    runner.update_budget = 0
    runner._cleanup = False
    runner._accrue_updates(2000, 3000)
    assert runner.update_credit == 0 and runner.update_budget == 0
    assert runner._can_update() is False
    with pytest.raises(RuntimeError, match="updates are forbidden"):
        runner._one_update(overlapped=True)
