"""Three real HTTP ports, synthetic physics/encoder, and real CPU SAC updates."""
from contextlib import contextmanager
from copy import deepcopy
import json
import threading

import numpy as np
import pytest
import torch

from pressb.online_rl.protocol import pose8_to_pose9, pose9_to_pose8
from pressb.online_rl.rpc import RPCClient, RPCServer
from pressb.online_rl.runner import OnlineRunner, RunConfig


TOKEN = "runner-integration-test"
CHECKPOINT_SHA = "a" * 64


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@contextmanager
def serve(handlers):
    server = RPCServer(("127.0.0.1", 0), handlers, token=TOKEN)
    thread = threading.Thread(target=server.serve_forever,
                              kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        assert not thread.is_alive()


class SyntheticSimulation:
    """One early success and one longer timeout; ended states never autoreset."""

    def __init__(self):
        self.resets, self.steps, self.closed = [], [], []
        self.after_step = None
        self.bad_discount = False
        self.early_success = True
        self.items = []

    def health(self, _):
        return dict(service="simulation", ready=True, action_horizon=7,
                    num_envs=2, control_state_dim=1, single_gamma=.99,
                    scene_sha256="synthetic-scene", config_sha256="synthetic-config",
                    collection_fingerprint="synthetic-collection",
                    robot_urdf_sha256="synthetic-robot", smoothing_window=1,
                    max_seconds=15., panel_bounds=dict(min_offset_x_m=-.01,
                    max_offset_x_m=.01, min_offset_y_m=-.02, max_offset_y_m=.02))

    def observation(self, eid, tick):
        marker = (len(self.resets) - 1) * 100 + eid * 10 + tick
        return dict(task=f"Press {self.resets[-1]['episodes'][eid]['floor']} floor.",
                    state=[marker / 1000., 0., .2, 1., 0., 0., 0., .008],
                    control_state=[float(marker)], images={})

    def reset(self, payload):
        assert not self.items or all(x["terminated"] or x["truncated"] for x in self.items)
        assert len(payload["episodes"]) == 2
        self.resets.append(deepcopy(payload))
        self.step_id = 0
        self.cohort_id = f"cohort-{len(self.resets)}"
        self.items = [dict(env_id=eid, episode_id=f"{self.cohort_id}-{eid}",
                           observation=self.observation(eid, 0), terminated=False,
                           truncated=False, info={}) for eid in range(2)]
        return dict(cohort_id=self.cohort_id, step_id=0,
                    items=deepcopy(self.items), all_done=False)

    def step(self, payload):
        assert payload["cohort_id"] == self.cohort_id
        assert payload["step_id"] == self.step_id
        active = [x["env_id"] for x in self.items if not (x["terminated"] or x["truncated"])]
        assert [x["env_id"] for x in payload["actions"]] == active
        for action in payload["actions"]:
            values = np.asarray(action["actions_pose8"])
            assert values.shape == (7, 8) and np.isfinite(values).all()
            np.testing.assert_allclose(values[:, 7], .008)
            np.testing.assert_allclose(np.linalg.norm(values[:, 3:7], axis=1), 1.)
        self.steps.append(deepcopy(payload))
        self.step_id += 1
        for eid, item in enumerate(self.items):
            if eid not in active:
                item.update(reward=0., executed_physics_steps=0, discount=1.)
                continue
            success = eid == 0 and (self.early_success or self.step_id == 2)
            ticks = (3 if success else 28) if eid == 0 else (28 if self.step_id == 1 else 5)
            item.update(observation=self.observation(eid, self.step_id),
                        terminated=success, truncated=eid == 1 and self.step_id == 2,
                        reward=.99 ** ((ticks - 1) / 4) if success else 0.,
                        discount=.99 ** (ticks / 4), executed_physics_steps=ticks,
                        info=dict(success=success, outcome="success" if success else "timeout"))
        if self.bad_discount:
            self.items[0]["discount"] = .1
        result = dict(cohort_id=self.cohort_id, step_id=self.step_id,
                      items=deepcopy(self.items),
                      all_done=all(x["terminated"] or x["truncated"] for x in self.items))
        if self.after_step:
            self.after_step(self.step_id)
        return result

    def close_run(self, payload):
        assert all(x["terminated"] or x["truncated"] for x in self.items)
        self.closed.append(deepcopy(payload))
        return {"closed": True}

    @property
    def handlers(self):
        return {"/health": self.health, "/reset": self.reset,
                "/step": self.step, "/close_run": self.close_run}


class SyntheticInference:
    def __init__(self):
        self.contexts, self.encoded, self.decoded, self.released = {}, [], [], []
        self.counter = 0
        self.invalid_feature = False
        self.global_rng_seed = None
        self.round_wire_actions = False

    def health(self, _):
        return dict(service="inference", ready=True, action_horizon=7,
                    feature_dim=2048, feature_schema="mean_embodied_action_tokens",
                    source_identity="synthetic-frozen-source", checkpoint_step=7,
                    checkpoint_sha256=CHECKPOINT_SHA, checkpoint_verified=True)

    def encode(self, payload):
        if self.global_rng_seed is not None:
            torch.manual_seed(self.global_rng_seed)
        result = []
        assert len(payload["observations"]) == len(payload["seeds"])
        for observation, seed in zip(payload["observations"], payload["seeds"], strict=True):
            self.counter += 1
            context_id = f"context-{self.counter}"
            feature = np.zeros(2048, np.float32)
            feature[0] = observation["control_state"][0]
            state9 = pose8_to_pose9(observation["state"])
            item = dict(context_id=context_id, feature=feature.tolist(),
                        state_pose9=state9.tolist())
            if payload["include_base_actions"]:
                base = np.tile(state9, (7, 1))
                base[:, 1] += np.random.default_rng(seed).normal(0, .001, 7)
                item.update(base_actions_pose9=base.tolist(),
                            base_actions_pose8=pose9_to_pose8(base).tolist())
                if self.round_wire_actions:
                    item["base_actions_pose8"] = np.asarray(item["base_actions_pose8"], dtype=np.float32).tolist()
            self.contexts[context_id] = deepcopy(item)
            self.encoded.append(dict(item=deepcopy(item), seed=seed))
            result.append(item)
        if self.invalid_feature:
            result[0]["feature"] = [0.]
        return {"items": result}

    def decode(self, payload):
        result = []
        for item in payload["items"]:
            context = self.contexts[item["context_id"]]
            noise = np.asarray(item["initial_noise"])
            assert noise.shape == (7, 9)
            np.testing.assert_allclose(noise, np.tile(noise[0], (7, 1)))
            assert np.abs(noise).max() <= 1.5 + 1e-6
            actions = np.tile(context["state_pose9"], (7, 1))
            actions[:, :3] += .001 * noise[:, :3]
            result.append(dict(context_id=item["context_id"],
                               actions_pose9=actions.tolist(),
                               actions_pose8=pose9_to_pose8(actions).tolist()))
            self.decoded.append(deepcopy(item))
        return {"items": result}

    def release(self, payload):
        for cid in payload["context_ids"]:
            assert cid in self.contexts, "Context released twice or never encoded"
            del self.contexts[cid]
            self.released.append(cid)
        return {"released": len(payload["context_ids"])}

    @property
    def handlers(self):
        return {"/health": self.health, "/encode": self.encode,
                "/decode": self.decode, "/release": self.release}


def config(method, **overrides):
    values = dict(method=method, max_transitions=3, eval_episodes=2,
                  learning_start=0, training_frequency=1, checkpoint_every=2,
                  progressive_exploration=0, replay_capacity=32,
                  expected_checkpoint_step=7, expected_checkpoint_sha256=CHECKPOINT_SHA,
                  learner=dict(hidden_dim=16, batch_size=2, num_qs=2, policy_freq=1))
    values.update(overrides)
    return RunConfig(**values)


@contextmanager
def three_nodes(tmp_path, cfg, *, checkpoint=None, resume=False):
    simulation, inference = SyntheticSimulation(), SyntheticInference()
    with serve(simulation.handlers) as sim_url, serve(inference.handlers) as inference_url:
        runner = OnlineRunner(cfg, sim_url, inference_url, tmp_path, token=TOKEN,
                              checkpoint=checkpoint, resume=resume, device="cpu", timeout=5)
        with serve({"/health": runner.health, "/status": runner.health,
                    "/stop": runner.stop}) as learner_url:
            assert len({sim_url, inference_url, learner_url}) == 3
            client = RPCClient(learner_url, token=TOKEN, timeout=5)
            assert client.call("/health")["service"] == "learner"
            yield runner, simulation, inference, client


def assert_completed(runner, simulation, inference, client):
    assert len(simulation.resets) == 1
    assert len(simulation.steps) == 2
    assert len(simulation.closed) == 1
    assert simulation.closed[0]["run_id"] == runner.run_id
    assert not inference.contexts and not runner.contexts
    assert len(inference.released) == len(inference.encoded) == 5
    assert len(set(inference.released)) == 5
    status = client.call("/status")
    assert status["episodes"] == 2
    assert status["successes"] == 1
    assert status["transitions"] == 3
    assert status["physical_ticks"] == 36
    assert status["control_steps"] == 9
    assert status["success_rate"] == .5
    assert not status["ready"]
    persisted = json.loads((runner.output / "status.json").read_text())
    assert {**persisted, "protocol_version": 1} == status


@pytest.mark.parametrize("method", ["action_residual", "initial_noise"])
def test_train_then_evaluate_checkpoint_over_three_http_ports(tmp_path, method):
    training = tmp_path / "train"
    with three_nodes(training, config(method)) as (runner, simulation, inference, client):
        result = runner.run()
        assert result["state"] == "complete"
        assert_completed(runner, simulation, inference, client)
        assert result["updates"] == 3
        replay = runner.replay.state_dict()["arrays"]
        np.testing.assert_array_equal(replay["obs"][:, 0], [0, 10, 11])
        np.testing.assert_array_equal(replay["next_obs"][:, 0], [1, 11, 12])
        np.testing.assert_allclose(replay["discount"][:, 0], .99 ** (np.array([3, 28, 5]) / 4))
        np.testing.assert_allclose(replay["reward"][:, 0], [.99 ** .5, 0, 0])
        np.testing.assert_array_equal(replay["terminated"][:, 0], [True, False, False])
        np.testing.assert_array_equal(replay["truncated"][:, 0], [False, False, True])
        if method == "action_residual":
            np.testing.assert_allclose(replay["next_base_action"][:, 0], [.001, .011, .012])
            assert not inference.decoded
        else:
            assert "base_action" not in replay
            assert len(inference.decoded) == 3
            for row, decoded in zip(replay["action"], inference.decoded, strict=True):
                np.testing.assert_allclose(decoded["initial_noise"], np.tile(row * 1.5, (7, 1)))
        saved_actor = deepcopy(runner.learner.actor.state_dict())
        metrics = [json.loads(line) for line in (training / "metrics.jsonl").read_text().splitlines()]
        assert all(item["actor_updated"] == 1 for item in metrics)
        assert (training / "last.pt").is_file()
        assert (training / "step_00000002.pt").is_file()

    evaluation = tmp_path / "evaluate"
    with three_nodes(evaluation, config(method, mode="eval"),
                     checkpoint=training / "last.pt") as nodes:
        runner, simulation, inference, client = nodes
        runner.run()
        assert_completed(*nodes)
        assert runner.learner.updates == 3
        assert len(runner.replay) == 0
        assert not (evaluation / "last.pt").exists()
        for name, value in runner.learner.actor.state_dict().items():
            torch.testing.assert_close(value, saved_actor[name], rtol=0, atol=0)


def test_resume_preserves_replay_and_counters_at_fresh_cohort(tmp_path):
    with three_nodes(tmp_path / "initial", config("initial_noise")) as nodes:
        nodes[0].run()
    with three_nodes(tmp_path / "resume", config("initial_noise", max_transitions=6),
                     checkpoint=tmp_path / "initial" / "last.pt", resume=True) as nodes:
        runner, simulation, inference, client = nodes
        result = runner.run()
        assert result["transitions"] == 6 and result["updates"] == 6
        assert result["episodes"] == 4 and result["successes"] == 2
        assert result["physical_ticks"] == 72 and result["control_steps"] == 18
        assert len(runner.replay) == 6
        assert len(simulation.resets) == len(simulation.closed) == 1
        assert not inference.contexts
        assert simulation.resets[0]["episodes"][0]["floor"] == 26
        event = json.loads((runner.output / "events.jsonl").read_text().splitlines()[0])
        assert event["event"] == "resume" and event["exact_physical_resume"] is False
        assert client.call("/status")["state"] == "complete"


def test_residual_warmup_executes_original_wire_actions_exactly(tmp_path):
    with three_nodes(tmp_path / "warmup", config("action_residual", learning_start=100)) as nodes:
        runner, simulation, inference, _ = nodes
        inference.round_wire_actions = True
        runner.run()
        for action, encoded in zip(simulation.steps[0]["actions"], inference.encoded[:2], strict=True):
            np.testing.assert_array_equal(action["actions_pose8"], encoded["item"]["base_actions_pose8"])
        np.testing.assert_array_equal(simulation.steps[1]["actions"][0]["actions_pose8"],
                                      inference.encoded[3]["item"]["base_actions_pose8"])
        assert not runner.replay.state_dict()["arrays"]["action"].any()
        assert runner.learner.updates == 0


def test_frozen_baseline_uses_seeded_base_actions_without_learner(tmp_path):
    with three_nodes(tmp_path / "base", config("base", mode="eval")) as nodes:
        runner, simulation, inference, client = nodes
        runner.run()
        assert_completed(*nodes)
        assert runner.learner is None and runner.replay is None
        assert not inference.decoded
        for action, encoded in zip(simulation.steps[0]["actions"], inference.encoded[:2], strict=True):
            np.testing.assert_array_equal(action["actions_pose8"], encoded["item"]["base_actions_pose8"])
        assert runner.health()["updates"] == 0
        assert not (runner.output / "last.pt").exists()


def test_http_stop_finishes_active_cohort_and_saves_checkpoint(tmp_path):
    with three_nodes(tmp_path / "stop", config("initial_noise", max_transitions=100)) as nodes:
        runner, simulation, inference, client = nodes
        accepted = []
        simulation.after_step = lambda step: accepted.append(client.call("/stop", {})) if step == 1 else None
        result = runner.run()
        assert_completed(*nodes)
        assert result["state"] == "stopped"
        assert len(accepted) == 1 and accepted[0]["accepted"]
        assert (runner.output / "last.pt").is_file()


def test_invalid_duration_fails_without_fabricating_transition_and_releases_contexts(tmp_path):
    with three_nodes(tmp_path / "failed", config("initial_noise")) as nodes:
        runner, simulation, inference, client = nodes
        simulation.bad_discount = True
        with pytest.raises(ValueError, match="actual physical duration"):
            runner.run()
        assert client.call("/status")["state"] == "failed"
        assert len(runner.replay) == 0
        assert not inference.contexts and not runner.contexts
        assert len(inference.released) == len(inference.encoded) == 4
        assert not simulation.closed  # Still-active physical cohort must remain owned.


def test_paired_evaluation_seeds_survive_other_environment_ending_early(tmp_path):
    seeds = []
    for early_success in (True, False):
        with three_nodes(tmp_path / str(early_success), config("base", mode="eval")) as nodes:
            runner, simulation, inference, _ = nodes
            simulation.early_success = early_success
            runner.run()
            seeds.append([encoded["seed"] for encoded in inference.encoded
                          if encoded["item"]["feature"][0] in (10, 11, 12)])
            assert not inference.contexts and len(simulation.closed) == 1
    assert len(seeds[0]) == 3 and seeds[0] == seeds[1]


def test_invalid_first_encoded_sample_releases_whole_returned_batch(tmp_path):
    with three_nodes(tmp_path / "bad-encoding", config("base", mode="eval")) as nodes:
        runner, simulation, inference, client = nodes
        inference.invalid_feature = True
        with pytest.raises(ValueError, match="encoder feature"):
            runner.run()
        assert client.call("/status")["state"] == "failed"
        assert len(inference.released) == len(inference.encoded) == 2
        assert not inference.contexts and not runner.contexts
        assert not simulation.steps and not simulation.closed


def test_resume_rejects_changed_update_schedule_before_reset(tmp_path):
    with three_nodes(tmp_path / "initial", config("initial_noise")) as nodes:
        nodes[0].run()
    for changed in ({"utd": .5}, {"training_frequency": 2},
                    {"learning_start": 1}, {"progressive_exploration": 50}):
        with three_nodes(tmp_path / next(iter(changed)),
                         config("initial_noise", max_transitions=6, **changed),
                         checkpoint=tmp_path / "initial" / "last.pt", resume=True) as nodes:
            runner, simulation, inference, client = nodes
            with pytest.raises(ValueError, match="sampling/update schedule"):
                runner.run()
            assert client.call("/status")["state"] == "failed"
            assert not simulation.resets and not inference.encoded


def test_noise_eval_seed_controls_sampling_independently_of_global_rng_and_batch(tmp_path):
    with three_nodes(tmp_path / "initial", config("initial_noise")) as nodes:
        nodes[0].run()
    samples = []
    for index, (eval_seed, global_rng_seed, early_success) in enumerate(
            [(42, 111, True), (42, 222, False), (43, 222, False)]):
        with three_nodes(tmp_path / f"eval-{index}", config("initial_noise", mode="eval", seed=eval_seed),
                         checkpoint=tmp_path / "initial" / "last.pt") as nodes:
            runner, simulation, inference, _ = nodes
            simulation.early_success = early_success
            # This changes the process RNG after checkpoint loading and before
            # every action batch, so checkpoint RNG restoration cannot hide it.
            inference.global_rng_seed = global_rng_seed
            runner.run()
            markers = {row["item"]["context_id"]: row["item"]["feature"][0]
                       for row in inference.encoded}
            samples.append({markers[row["context_id"]]: np.asarray(row["initial_noise"])
                            for row in inference.decoded})
            assert not inference.contexts and len(simulation.closed) == 1
    assert set(samples[0]) == {0, 10, 11}
    for marker in samples[0]:
        np.testing.assert_array_equal(samples[0][marker], samples[1][marker])
        assert not np.array_equal(samples[1][marker], samples[2][marker])
