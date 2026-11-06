"""Real port-to-port pipeline tests; fixtures are explicitly not Isaac/model benchmarks."""
from contextlib import contextmanager
import threading

import numpy as np
import pytest
import torch

from pressb.runtime.rpc import RPCServer
from pressb.runtime.inference import InferenceService
from pressb.runtime.simulation import SimulationService
from pressb.runtime.trainer import GenericTrainer


def schemas(horizon, action_dim, noise_dim, kind):
    vector = {"names": [f"joint{i}" for i in range(action_dim)], "units": ["rad"] * action_dim}
    result = {"schema_version": 2, "node_type": kind,
              "observation": {"state": vector, "cameras": []},
              "action": {**vector, "horizon": horizon, "control_hz": 30,
                         "representation": "joint_position", "frame": "robot_base"}}
    if kind == "inference":
        result.update(model_id="synthetic-flow-fixture", feature_dim=action_dim,
                      capabilities=["encode", "decode", "features", "context_cache", "external_initial_noise"],
                      noise={"shape": [horizon, noise_dim]})
    else:
        result.update(task_id="synthetic-plant-fixture", num_envs=2, control_state_dim=action_dim,
                      task_features_dim=0, reset_mode="indexed", discount={"gamma": .99, "timebase_hz": 30})
    return result


class Flow:
    def __init__(self, horizon, action_dim, noise_dim):
        self.h, self.d, self.n = horizon, action_dim, noise_dim

    def describe(self):
        return schemas(self.h, self.d, self.n, "inference")

    def encode_batch(self, observations, seeds):
        return [((np.asarray(obs["state"]), seed), np.asarray(obs["state"]))
                for obs, seed in zip(observations, seeds)]

    def decode_batch(self, contexts, noises):
        results = []
        for (state, seed), z in zip(contexts, noises):
            z = np.random.default_rng(seed).normal(size=(self.h, self.n)) if z is None else z
            results.append(.5 * state[None] + .05 * z[:, :self.d])
        return results


class Plant:
    def __init__(self, horizon, action_dim, noise_dim, fused=False):
        self.h, self.d, self.n = horizon, action_dim, noise_dim
        self.fused = fused
        self.state = np.zeros((2, action_dim))
        self.tick = [0, 0]
        self.resets = [0, 0]
        self.steps = 0

    def describe(self):
        result = schemas(self.h, self.d, self.n, "simulation")
        if self.fused:
            result.update(capabilities=["terminal_state_only", "step_and_reset"],
                          terminal_observation_modes=["full", "state_only_if_no_bootstrap"])
        return result

    def _obs(self, i):
        return dict(state=self.state[i].tolist(), control_state=self.state[i].tolist(),
                    task_features=[], images={}, task="Reach a target.")

    def reset(self, env_ids, seed, options):
        for i in env_ids:
            self.state[i] = np.random.default_rng(seed+i).uniform(-.1, .1, self.d)
            self.tick[i] = 0
            self.resets[i] += 1
        return [dict(env_id=i, observation=self._obs(i)) for i in env_ids]

    def step(self, items, *, terminal_observation_mode="full"):
        results = []
        for item in items:
            i = item["env_id"]
            self.state[i] = np.asarray(item["actions"])[-1]
            self.tick[i] += 1
            self.steps += 1
            done = self.tick[i] >= 2+i
            results.append(dict(env_id=i, observation=self._obs(i),
                                reward=float(-np.linalg.norm(self.state[i])), discount=.99 ** len(item["actions"]),
                                terminated=done, truncated=False,
                                info={"elapsed_control_steps": len(item["actions"]),
                                      "physics_steps": 4*len(item["actions"]), "success": bool(done)}))
            if done and terminal_observation_mode == "state_only_if_no_bootstrap":
                results[-1]["terminal_observation_omitted"] = True
        return results

    def step_and_reset(self, items, *, reset_seed, reset_options, max_active_envs):
        rows = self.step(items, terminal_observation_mode="state_only_if_no_bootstrap")
        done = [row["env_id"] for row in rows if row["terminated"] or row["truncated"]]
        reset_ids = done[:max(0, max_active_envs - (len(items) - len(done)))]
        resets = self.reset(reset_ids, reset_seed, reset_options) if reset_ids else []
        return {"items": rows, "reset_items": resets}


@contextmanager
def endpoint(service):
    with RPCServer(("127.0.0.1", 0), service.handlers) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .02}, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join(3)


@pytest.mark.parametrize("horizon,action_dim,noise_dim", [(7, 9, 9), (32, 7, 20)])
@pytest.mark.parametrize("method", ["dsrl", "action_residual"])
@pytest.mark.parametrize("fused", [False, True])
def test_real_http_collect_update_checkpoint_resume(tmp_path, horizon, action_dim, noise_dim, method, fused):
    torch.set_num_threads(1)
    inference = InferenceService(Flow(horizon, action_dim, noise_dim), batch_size=2, cache_size=8)
    plant = Plant(horizon, action_dim, noise_dim, fused=fused)
    simulation = SimulationService(plant)
    learner = dict(hidden_dim=16, batch_size=2, noise_steps=horizon, residual_mode="all", num_qs=2,
                   residual_scale=[.02] * action_dim)
    with endpoint(inference) as inf, endpoint(simulation) as sim:
        config = dict(method=method, output_dir=str(tmp_path / "first"), learner=learner,
                      endpoints={"inference": inf, "simulation": sim},
                      rollout={"max_transitions": 10, "execution_horizon": 2},
                      update={"learning_starts": 0, "replay_capacity": 32, "utd": 1},
                      checkpoint={"interval": 4, "keep": 2, "include_replay": True})
        if fused:
            config["rollout"].update(terminal_observation_mode="state_only_if_no_bootstrap", fused_step_reset=True)
        runner = GenericTrainer(config)
        report = runner.run()
        assert report["transitions"] == 10 and report["updates"] > 0
        assert plant.steps == 10 and min(plant.resets) >= 2
        assert not inference.contexts and simulation.run_id is None
        assert report["reset_envs"] == sum(plant.resets)
        if fused:
            assert report["rpc_calls"]["simulation/reset"] == 1
            assert report["rpc_calls"]["simulation/step_and_reset"] > 0
            assert "simulation/step" not in report["rpc_calls"]
        assert runner.replay.arrays["action"].shape[1] == horizon * (noise_dim if method == "dsrl" else action_dim)
        saved = report["last_checkpoint"]
        config["output_dir"] = str(tmp_path / "resumed")
        config["rollout"]["max_transitions"] = 14
        config["checkpoint"]["resume"] = saved
        resumed = GenericTrainer(config)
        resumed_report = resumed.run()
        assert resumed_report["transitions"] == 14 and len(resumed.replay) == 14
        assert resumed_report["updates"] >= report["updates"]
        assert not inference.contexts and simulation.run_id is None
