"""Capability-negotiated DPPO resets preserve per-episode random streams."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from test_dppo_training import make_runner
from test_rl_nodes_trainer import Simulation


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class SeededSimulation(Simulation):
    """Exercise seed-dependent observations; return reset rows out of order."""

    def __init__(self, metadata):
        super().__init__(metadata)
        self.effective_seeds = {}

    def observation(self, eid):
        result = super().observation(eid)
        rng = np.random.default_rng(np.random.SeedSequence([self.effective_seeds[eid], eid]))
        result["state"][0] = float(rng.uniform()) + self.slots[eid]["step_id"] / 100
        return result

    def call(self, path, payload=None):
        if path == "/reset":
            overrides = payload.get("env_seeds", {})
            for eid in payload["env_ids"]:
                self.effective_seeds[eid] = overrides.get(str(eid), payload["seed"])
        result = super().call(path, payload)
        if path == "/reset":
            for row in result["items"]:
                row["info"]["effective_seed"] = self.effective_seeds[row["env_id"]]
            result["items"].reverse()
        return result


class Observer:
    def __init__(self):
        self.resets, self.transitions = [], []

    def on_reset(self, rows):
        self.resets.append(deepcopy(rows))
        rows[0]["observation"]["state"][0] = -999  # callbacks receive a copy

    def on_transition(self, before, actions, after):
        self.transitions.append((before, actions, after))


def reset_runner(output, *, capable=True, initial_all=False):
    output.mkdir(exist_ok=True)
    runner, inference, simulation = make_runner(output)
    simulation = SeededSimulation(deepcopy(simulation.meta))
    if capable:
        simulation.meta["capabilities"] = ["per_env_reset_seeds"]
    if initial_all:
        simulation.meta["initial_reset"] = "all_envs"
    runner.simulation = simulation
    runner.simulation_metadata = deepcopy(simulation.meta)
    runner.observer = Observer()
    return runner, inference, simulation


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_mixed_episode_unsorted_batch_preserves_legacy_seed_observation_and_logs(tmp_path):
    batched, _, new_sim = reset_runner(tmp_path / "batched")
    legacy, _, old_sim = reset_runner(tmp_path / "legacy", capable=False)
    for runner in (batched, legacy):
        runner.episode_serial.update({0: 7, 1: 3})
        runner.config.rollout.reset_options = {"task": "press the selected button"}
        runner._reset([1, 0])

    assert [request["env_ids"] for request in new_sim.resets] == [[1, 0]]
    assert [request["env_ids"] for request in old_sim.resets] == [[1], [0]]
    expected = {eid: batched._seed(eid, serial, "reset") for eid, serial in ((1, 3), (0, 7))}
    request = new_sim.resets[0]
    assert request["seed"] == expected[1]
    assert request["env_seeds"] == {str(eid): seed for eid, seed in expected.items()}
    assert request["options"] == batched.config.rollout.reset_options
    assert all("env_seeds" not in request for request in old_sim.resets)
    assert batched.episode_seed == legacy.episode_seed == expected
    assert batched.slots == legacy.slots
    assert not batched._physics_in_flight
    assert batched._claimed
    by_id = lambda rows: {row["env_id"]: row for row in rows}
    logs = by_id(read_jsonl(batched.output / "resets.jsonl"))
    assert logs == by_id(read_jsonl(legacy.output / "resets.jsonl"))
    for eid, row in logs.items():
        assert row["seed"] == row["info"]["effective_seed"] == expected[eid]
        assert row["policy_seed_at_step0"] == batched._seed(eid, row["episode_index"], "policy", 0)
    assert len(batched.observer.resets) == 1
    assert by_id(batched.observer.resets[0]) == batched.slots
    assert "env_seeds keyed by env_id" in batched._seed_schedule()["subsequent_reset"]
    assert legacy._seed_schedule()["subsequent_reset"].startswith("one environment per request")


@pytest.mark.parametrize("capable", [False, True])
@pytest.mark.parametrize("initial_all", [False, True])
def test_initial_reset_keeps_shared_or_indexed_rule(tmp_path, capable, initial_all):
    runner, _, simulation = reset_runner(tmp_path, capable=capable, initial_all=initial_all)
    runner._reset([0, 1], initial=True)
    assert [row["env_ids"] for row in simulation.resets] == ([[0, 1]] if initial_all else [[0], [1]])
    assert all("env_seeds" not in row for row in simulation.resets)
    expected = {eid: runner._seed(0 if initial_all else eid, 0, "reset") for eid in (0, 1)}
    assert runner.episode_seed == simulation.effective_seeds == expected
    assert {row["env_id"]: row["seed"] for row in read_jsonl(tmp_path / "resets.jsonl")} == expected


@pytest.mark.parametrize("capable", [False, True])
def test_singleton_and_empty_reset(tmp_path, capable):
    runner, _, simulation = reset_runner(tmp_path, capable=capable)
    runner._reset([])
    assert not simulation.resets and not runner._claimed
    runner.episode_serial[1] = 9
    runner._reset([1])
    assert len(simulation.resets) == 1
    request = simulation.resets[0]
    assert request["seed"] == runner._seed(1, 9, "reset")
    if capable:
        assert request["env_seeds"] == {"1": request["seed"]}
    else:
        assert "env_seeds" not in request


@pytest.mark.parametrize("capable", [False, True])
def test_reset_response_loss_is_not_retried(tmp_path, monkeypatch, capable):
    runner, _, simulation = reset_runner(tmp_path, capable=capable)
    original = simulation.call

    def lose_response(path, payload=None):
        original(path, payload)
        raise TimeoutError("reset already advanced the physical state")

    monkeypatch.setattr(simulation, "call", lose_response)
    with pytest.raises(TimeoutError, match="already advanced"):
        runner._reset([1, 0])
    assert len(simulation.resets) == 1
    assert runner._physics_in_flight and runner._claimed
    assert not runner.slots and not runner.observer.resets


@pytest.mark.parametrize("invalid", ["missing_row", "duplicate_env", "duplicate_episode", "nonzero_step"])
def test_batch_response_keeps_identity_validation(tmp_path, monkeypatch, invalid):
    runner, _, simulation = reset_runner(tmp_path)
    original = simulation.call

    def corrupt(path, payload=None):
        result = original(path, payload)
        rows = result["items"]
        if invalid == "missing_row":
            rows.pop()
        elif invalid == "duplicate_env":
            rows[1]["env_id"] = rows[0]["env_id"]
        elif invalid == "duplicate_episode":
            rows[1]["episode_id"] = rows[0]["episode_id"]
        else:
            rows[0]["step_id"] = 1
        return result

    monkeypatch.setattr(simulation, "call", corrupt)
    with pytest.raises(ValueError, match="incorrect environment ids|unique episode_id and step zero"):
        runner._reset([1, 0])
    assert len(simulation.resets) == 1
    assert not runner.observer.resets


def test_real_collect_batch_reset_preserves_terminal_bootstrap_policy_and_gae(tmp_path):
    outcomes = []
    for name, capable in (("legacy", False), ("batched", True)):
        runner, inference, simulation = reset_runner(tmp_path / name, capable=capable, initial_all=True)
        runner.config.dppo.rollout_steps = 4
        runner._negotiate()
        runner._reset([0, 1], initial=True)
        records = runner._collect()
        runner._advantages(records)
        outcomes.append((runner, inference, simulation, records))

    legacy, old_inference, old_sim, old_records = outcomes[0]
    batched, new_inference, new_sim, new_records = outcomes[1]
    # Env 0 terminates every step; env 1 truncates at step 3, then starts anew.
    assert [row["env_ids"] for row in new_sim.resets] == [[0, 1], [0], [0], [0, 1], [0]]
    assert [row["env_ids"] for row in old_sim.resets] == [[0, 1], [0], [0], [0], [1], [0]]
    simultaneous = new_sim.resets[3]
    assert simultaneous["env_seeds"] == {"0": batched._seed(0, 3, "reset"),
                                          "1": batched._seed(1, 1, "reset")}
    assert batched.transitions == legacy.transitions == 8
    assert batched.episodes == legacy.episodes == 5
    assert batched.episode_serial == legacy.episode_serial == {0: 4, 1: 1}
    assert batched.episode_seed == legacy.episode_seed
    assert batched.slots == legacy.slots
    scalar_keys = ("env_id", "episode_id", "step_id", "seed", "reward", "discount", "value",
                   "next_value", "terminated", "truncated", "advantage", "return")
    for old, new in zip(old_records, new_records, strict=True):
        assert {key: old[key] for key in scalar_keys} == {key: new[key] for key in scalar_keys}
        assert torch.equal(old["chain"], new["chain"])
        assert torch.equal(old["logprobs"], new["logprobs"])
        np.testing.assert_array_equal(old["actions"], new["actions"])
        if new["terminated"]:
            assert new["next_value"] == 0.
            assert new["return"] == pytest.approx(new["reward"])
        elif new["truncated"]:
            assert new["return"] == pytest.approx(new["reward"] + new["discount"] * new["next_value"])
    # Both policies and time-limit bootstrapping see the same observations/seeds.
    for route in ("/sample_chain", "/encode"):
        old_requests = [payload for name, payload in old_inference.calls if name == route]
        new_requests = [payload for name, payload in new_inference.calls if name == route]
        assert old_requests == new_requests
    assert read_jsonl(legacy.output / "episodes.jsonl") == read_jsonl(batched.output / "episodes.jsonl")
    assert len(batched.observer.transitions) == 4
    assert new_records[5]["truncated"] and new_records[7]["step_id"] == 0
    assert new_records[5]["episode_id"] != new_records[7]["episode_id"]
