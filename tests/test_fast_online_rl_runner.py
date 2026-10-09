"""Real HTTP and CPU SAC tests for indexed reset / pipeline accounting."""
from contextlib import contextmanager
from copy import deepcopy
import base64
import json
import threading
import time

import numpy as np
import pytest
import torch

from pressb.online_rl.fast_runner import FastOnlineRunner, FastRunConfig, RESET_MODE
from pressb.online_rl.rpc import RPCClient
from test_online_rl_runner import (CHECKPOINT_SHA, TOKEN, SyntheticInference,
                                  SyntheticSimulation, serve, three_nodes as legacy_nodes,
                                  config as legacy_config)


INFERENCE_SOURCE = dict(source_sha256={"starVLA/inference/piper_policy.py": "f" * 64},
                        addon_sha256="a" * 64, vla_repo="/original/repository")


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class IndexedSimulation(SyntheticSimulation):
    """Env 0 succeeds each chunk while env 1 takes three chunks to timeout."""
    def __init__(self):
        super().__init__()
        self.indexed_resets = []
        self.serial = [0, 0]
        self.ages = [0, 0]
        self.layouts = {}
        self.delay = 0.
        self.image_resolution = [224, 224]
        self.sources = {"src/pressb/online_rl/batched_control.py":
                        dict(path="/old-host/batched_control.py", bytes=100, sha256="b" * 64)}

    def health(self, payload):
        return {**super().health(payload), "reset_mode": RESET_MODE,
                "simulation_contract": "test-indexed-reset-v1",
                "image_resolution": self.image_resolution,
                "sources": self.sources,
                "control_backend": "synthetic", "renderer_settings": {"cadence": "chunk"}}

    def observation(self, eid, tick):
        marker = self.serial[eid] * 1000 + eid * 100 + tick
        return dict(task=f"Press {self.layouts[eid]['floor']} floor.",
                    state=[marker / 1000., 0., .2, 1., 0., 0., 0., .008],
                    control_state=[float(marker)], images={})

    def reset(self, payload):
        self.layouts = {eid: deepcopy(row) for eid, row in enumerate(payload["episodes"])}
        return super().reset(payload)

    def reset_envs(self, payload):
        assert payload["cohort_id"] == self.cohort_id and payload["step_id"] == self.step_id
        self.indexed_resets.append(deepcopy(payload))
        reset_items = []
        for row in payload["episodes"]:
            eid = row["env_id"]
            assert self.items[eid]["terminated"] or self.items[eid]["truncated"]
            self.serial[eid] += 1
            self.ages[eid] = 0
            self.layouts[eid] = deepcopy(row)
            item = dict(env_id=eid, episode_id=f"{self.cohort_id}-{eid}-reset{self.serial[eid]}",
                        observation=self.observation(eid, 0), terminated=False, truncated=False, info={})
            self.items[eid] = item
            reset_items.append(item)
        return dict(cohort_id=self.cohort_id, step_id=self.step_id,
                    items=deepcopy(reset_items), all_done=False)

    def step(self, payload):
        if self.delay:
            time.sleep(self.delay)
        assert payload["cohort_id"] == self.cohort_id and payload["step_id"] == self.step_id
        active = [item["env_id"] for item in self.items if not (item["terminated"] or item["truncated"])]
        assert [row["env_id"] for row in payload["actions"]] == active
        self.steps.append(deepcopy(payload))
        self.step_id += 1
        for eid, item in enumerate(self.items):
            if eid not in active:
                item.update(reward=0., discount=1., executed_physics_steps=0)
                continue
            self.ages[eid] += 1
            success = eid == 0
            timeout = eid == 1 and self.ages[eid] == 3
            ticks = 3 if success else 5 if timeout else 28
            item.update(observation=self.observation(eid, self.ages[eid]), terminated=success, truncated=timeout,
                        executed_physics_steps=ticks, discount=.99 ** (ticks / 4),
                        reward=.99 ** ((ticks - 1) / 4) if success else 0., info=dict(success=success))
        if self.bad_discount:
            self.items[active[0]]["discount"] = .1
        result = dict(cohort_id=self.cohort_id, step_id=self.step_id, items=deepcopy(self.items),
                      all_done=all(item["terminated"] or item["truncated"] for item in self.items))
        if self.after_step:
            self.after_step(self.step_id)
        return result

    @property
    def handlers(self):
        return {**super().handlers, "/reset_envs": self.reset_envs}


class BatchInference(SyntheticInference):
    def __init__(self):
        super().__init__()
        self.fused_calls = 0
        self.packed = False
        self.corrupt_packed = None
        self.source_identity = deepcopy(INFERENCE_SOURCE)

    def health(self, payload):
        return {**super().health(payload), "encode_release_context_ids": True,
                "source_identity": self.source_identity,
                "feature_encodings": ["json", "float32_base64"] if self.packed else ["json"]}

    def encode(self, payload):
        result = super().encode(payload)
        if payload.get("release_context_ids"):
            self.release({"context_ids": payload["release_context_ids"]})
            self.fused_calls += 1
        if payload.get("feature_encoding") == "float32_base64":
            for item in result["items"]:
                feature = np.asarray(item.pop("feature"), dtype="<f4")
                if self.corrupt_packed == "nonfinite":
                    feature[0] = np.nan
                item["feature_f32_b64"] = base64.b64encode(feature.tobytes()).decode("ascii")
                item["feature_dim"] = 2048
                if self.corrupt_packed == "dimension":
                    item["feature_dim"] = 2047
                elif self.corrupt_packed == "bytes":
                    item["feature_f32_b64"] = base64.b64encode(b"short").decode("ascii")
                elif self.corrupt_packed == "base64":
                    item["feature_f32_b64"] = "not-valid-base64!"
        return result


def config(method="action_residual", **changes):
    values = dict(method=method, max_transitions=6, eval_episodes=5, learning_start=0,
                  training_frequency=1, checkpoint_every=100, rolling_checkpoint_every=4,
                  progressive_exploration=0, replay_capacity=32,
                  expected_checkpoint_step=7, expected_checkpoint_sha256=CHECKPOINT_SHA,
                  metrics_every_updates=2, status_interval_seconds=.01,
                  learner=dict(hidden_dim=16, batch_size=2, num_qs=2, policy_freq=1))
    values.update(changes)
    return FastRunConfig(**values)


@contextmanager
def nodes(output, cfg=None, **kwargs):
    simulation, inference = IndexedSimulation(), BatchInference()
    with serve(simulation.handlers) as sim_url, serve(inference.handlers) as infer_url:
        runner = FastOnlineRunner(cfg or config(), sim_url, infer_url, output,
                                  token=TOKEN, timeout=5, device="cpu", **kwargs)
        with serve({"/health": runner.health, "/status": runner.health, "/stop": runner.stop}) as url:
            try:
                yield runner, simulation, inference, RPCClient(url, token=TOKEN)
            finally:
                runner.executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.parametrize("method", ["action_residual", "initial_noise"])
def test_independent_refill_preserves_terminal_replay_and_exact_utd(tmp_path, method):
    with nodes(tmp_path / method, config(method)) as (runner, simulation, inference, client):
        result = runner.run()
        assert result["state"] == "complete" and result["transitions"] == 6
        assert result["episodes"] == 4 and result["successes"] == 3
        assert result["updates"] == 6 and result["update_credit"] == 0
        assert len(simulation.resets) == 1 and len(simulation.indexed_resets) == 2
        assert all([row["env_id"] for row in request["episodes"]] == [0]
                   for request in simulation.indexed_resets)
        replay = runner.replay.state_dict()["arrays"]
        np.testing.assert_array_equal(replay["obs"][:, 0], [0, 100, 1000, 101, 2000, 102])
        np.testing.assert_array_equal(replay["next_obs"][:, 0], [1, 101, 1001, 102, 2001, 103])
        np.testing.assert_array_equal(replay["terminated"][:, 0], [True, False, True, False, True, False])
        np.testing.assert_array_equal(replay["truncated"][:, 0], [False, False, False, False, False, True])
        np.testing.assert_allclose(replay["discount"][:, 0], .99 ** (np.array([3, 28, 3, 28, 3, 5]) / 4))
        assert len(simulation.closed) == 1
        assert not inference.contexts and not runner.contexts and not runner.pending_releases
        assert inference.fused_calls >= 3
        assert len(inference.released) == len(inference.encoded)
        assert result["session_transitions"] == 6 and result["session_updates"] == 6
        assert result["transitions_per_second"] == pytest.approx(6 / result["wall_seconds"])
        assert result["timing_seconds"]["updates"] > 0
        assert result["timing_seconds"]["checkpoints"] > 0
        assert (runner.output / "last.pt").is_file()
        persisted = json.loads((runner.output / "status.json").read_text())
        assert {**persisted, "protocol_version": 1} == client.call("/status")


def test_pipeline_updates_stay_on_main_thread_and_overlap_rpc(tmp_path):
    with nodes(tmp_path / "pipeline") as (runner, simulation, _, _):
        simulation.delay = .04
        original = runner._init_learning
        main_thread = threading.get_ident()
        calls = []

        def initialize():
            original()
            update = runner.learner.update

            def checked(batch):
                calls.append(threading.get_ident())
                assert threading.get_ident() == main_thread
                return update(batch)

            runner.learner.update = checked

        runner._init_learning = initialize
        result = runner.run()
        assert calls == [main_thread] * 6
        assert result["timing_seconds"]["updates_during_rpc"] > 0


@pytest.mark.parametrize("pipeline", [False, True])
def test_warmup_boundary_and_fractional_utd_are_accounted_exactly(tmp_path, pipeline):
    with nodes(tmp_path / str(pipeline), config(learning_start=1, utd=.5,
               training_frequency=10, pipeline_updates=pipeline)) as (runner, *_):
        result = runner.run()
        assert result["earned_update_credit"] == 2.5
        assert result["updates"] == 2 and result["update_credit"] == .5


def test_stop_disables_refill_and_drains_longer_peer(tmp_path):
    with nodes(tmp_path / "stop", config(max_transitions=100)) as (runner, simulation, inference, client):
        simulation.after_step = lambda step: client.call("/stop", {}) if step == 1 else None
        result = runner.run()
        assert result["state"] == "stopped"
        assert result["transitions"] == result["updates"] == 4
        assert result["episodes"] == 2 and not simulation.indexed_resets
        assert [len(row["actions"]) for row in simulation.steps] == [2, 1, 1]
        assert len(simulation.closed) == 1 and not inference.contexts


def test_evaluation_budget_need_not_be_divisible_by_parallel_count(tmp_path):
    with nodes(tmp_path / "eval", config(method="base", mode="eval", eval_episodes=5)) as (runner, simulation, inference, _):
        result = runner.run()
        assert result["episodes"] == 5 and runner.scheduled_episodes == 5
        assert result["transitions"] == 7 and result["updates"] == 0
        assert [len(request["episodes"]) for request in simulation.indexed_resets] == [1, 1, 1]
        assert len(simulation.closed) == 1 and not inference.contexts


def test_resume_restores_replay_but_reports_session_delta_throughput(tmp_path):
    with nodes(tmp_path / "initial") as (runner, *_):
        runner.run()
    with nodes(tmp_path / "resume", config(max_transitions=12),
               checkpoint=tmp_path / "initial/last.pt", resume=True) as (runner, simulation, _, _):
        result = runner.run()
        assert result["transitions"] == result["updates"] == len(runner.replay) == 12
        assert result["session_transitions"] == result["session_updates"] == 6
        assert result["transitions_per_second"] == pytest.approx(6 / result["wall_seconds"])
        assert simulation.resets[0]["episodes"][0]["floor"] == 28


def test_resume_rejects_changed_camera_contract_before_sampling(tmp_path):
    with nodes(tmp_path / "initial") as (runner, *_):
        runner.run()
    with nodes(tmp_path / "changed", config(max_transitions=12),
               checkpoint=tmp_path / "initial/last.pt", resume=True) as (runner, simulation, inference, _):
        simulation.image_resolution = [640, 480]
        with pytest.raises(ValueError, match="different frozen model or simulator contract"):
            runner.run()
        assert not simulation.resets and not inference.encoded


def test_legacy_actor_warmstart_transfers_no_critic_optimizer_replay_or_counters(tmp_path):
    with legacy_nodes(tmp_path / "legacy", legacy_config("action_residual")) as (old, *_):
        old.run()
        actor = deepcopy(old.learner.actor.state_dict())
    source = tmp_path / "legacy/last.pt"
    # Give the synthetic legacy checkpoint the file-hash provenance supplied
    # by the real frozen-inference node (legacy test fixtures use a label).
    payload = torch.load(source, map_location="cpu", weights_only=False)
    payload["extra"]["identities"]["inference"]["source_identity"] = deepcopy(INFERENCE_SOURCE)
    torch.save(payload, source)
    with nodes(tmp_path / "forbidden", checkpoint=source, resume=True) as (runner, simulation, _, _):
        with pytest.raises(ValueError, match="simulator contract"):
            runner.run()
        assert not simulation.resets
    with nodes(tmp_path / "warmstart", config(learning_start=100), warmstart_actor=source) as (runner, *_):
        result = runner.run()
        assert result["transitions"] == len(runner.replay) == 6
        assert result["updates"] == 0
        assert not runner.learner.q_optimizer.state and not runner.learner.actor_optimizer.state
        for name, value in runner.learner.actor.state_dict().items():
            torch.testing.assert_close(value, actor[name], rtol=0, atol=0)
        assert runner.warmstart_provenance["parent_counters"]["transitions"] == 3
        assert runner.warmstart_provenance["restored"] == "actor only"
    with nodes(tmp_path / "continued", config(learning_start=100, max_transitions=12),
               checkpoint=tmp_path / "warmstart/last.pt", resume=True) as (runner, *_):
        runner.run()
        assert runner.warmstart_provenance["parent_counters"]["transitions"] == 3
        saved = torch.load(runner.output / "last.pt", map_location="cpu", weights_only=False)
        assert saved["extra"]["warmstart_actor"] == runner.warmstart_provenance


@pytest.mark.parametrize("failure", ["duration", "feature"])
def test_errors_release_all_contexts_without_resetting_active_episodes(tmp_path, failure):
    with nodes(tmp_path / failure) as (runner, simulation, inference, client):
        if failure == "duration":
            simulation.bad_discount = True
        else:
            inference.invalid_feature = True
        with pytest.raises(ValueError):
            runner.run()
        assert client.call("/status")["state"] == "failed"
        assert len(runner.replay) == 0
        assert not runner.contexts and not inference.contexts
        assert not simulation.closed and not simulation.indexed_resets


def test_archive_and_rolling_checkpoints_use_bounded_retention(tmp_path):
    with nodes(tmp_path / "checkpoints", config(checkpoint_every=6, rolling_checkpoint_every=2)) as (runner, *_):
        runner.run()
        assert {path.name for path in runner.output.glob("*.pt")} == {"last.pt", "step_00000006.pt"}
        checkpoint = torch.load(runner.output / "last.pt", map_location="cpu", weights_only=False)
        assert checkpoint["updates"] == 6
        assert checkpoint["extra"]["update_credit"] == 0
        assert checkpoint["extra"]["fast_runner"]["contract"] == "independent_episodes_pipeline_v1"


def test_packed_features_decode_exactly_and_fallback_remains_compatible(tmp_path):
    arrays = []
    for packed in (False, True):
        with nodes(tmp_path / str(packed), config(learning_start=100)) as (runner, _, inference, _):
            inference.packed = packed
            runner.run()
            assert runner.feature_encoding == ("float32_base64" if packed else "json")
            arrays.append(runner.replay.state_dict()["arrays"])
            assert not inference.contexts
    for name in arrays[0]:
        np.testing.assert_array_equal(arrays[0][name], arrays[1][name])


@pytest.mark.parametrize("corruption", ["dimension", "bytes", "base64", "nonfinite"])
def test_malformed_packed_features_fail_and_release_whole_batch(tmp_path, corruption):
    with nodes(tmp_path / corruption) as (runner, simulation, inference, _):
        inference.packed = True
        inference.corrupt_packed = corruption
        with pytest.raises(ValueError, match="Packed encoder feature"):
            runner.run()
        assert not runner.contexts and not inference.contexts and not simulation.steps
        assert len(inference.released) == 2


@pytest.mark.parametrize("changed_hash", [False, True])
def test_source_hash_resume_contract_ignores_host_paths_but_rejects_changed_code(tmp_path, changed_hash):
    with nodes(tmp_path / "initial") as (runner, *_):
        runner.run()
    with nodes(tmp_path / "resume", config(max_transitions=12), checkpoint=tmp_path / "initial/last.pt",
               resume=True) as (runner, simulation, _, _):
        source = simulation.sources["src/pressb/online_rl/batched_control.py"]
        source["path"] = "/other-host/project/batched_control.py"
        source["bytes"] = 200
        if changed_hash:
            source["sha256"] = "c" * 64
            with pytest.raises(ValueError, match="simulator contract"):
                runner.run()
            assert not simulation.resets
        else:
            assert runner.run()["transitions"] == 12


@pytest.mark.parametrize("source_change", ["addon_only", "core", "missing"])
def test_actor_transfer_checks_core_model_source_but_allows_adapter_update(tmp_path, source_change):
    with nodes(tmp_path / "initial") as (runner, *_):
        runner.run()
    with nodes(tmp_path / "transfer", config(learning_start=100),
               warmstart_actor=tmp_path / "initial/last.pt") as (runner, simulation, inference, _):
        inference.source_identity["addon_sha256"] = "d" * 64
        inference.source_identity["vla_repo"] = "/different/machine/repository"
        if source_change == "core":
            inference.source_identity["source_sha256"]["starVLA/inference/piper_policy.py"] = "e" * 64
        elif source_change == "missing":
            inference.source_identity = None
        if source_change == "addon_only":
            assert runner.run()["transitions"] == 6
        else:
            with pytest.raises(ValueError, match="core VLA source files"):
                runner.run()
            assert not simulation.resets
