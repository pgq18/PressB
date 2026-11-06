"""CPU SAC through the public three-node contract, without simulator dependencies."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import pressb.runtime.trainer as trainer_module
from pressb.runtime.trainer import GenericTrainer, TrainerConfig, _wandb_payload, _wandb_worker


@pytest.fixture(autouse=True)
def one_cpu_thread():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def metadata(horizon=32, action_dim=7, noise_dim=20, state_dim=7):
    action = dict(horizon=horizon, names=[f"action_{i}" for i in range(action_dim)],
                  units=["rad"] * (action_dim - 1) + ["m"], representation="joint_absolute",
                  frame="robot_base", control_hz=30.)
    obs = dict(state=dict(names=[f"state_{i}" for i in range(state_dim)],
                         units=["rad"] * (state_dim - 1) + ["m"]), cameras=[])
    inference = dict(schema_version=2, node_type="inference", model_id="fixed-test-weights",
                     action=action, observation=obs, noise={"shape": [horizon, noise_dim]},
                     feature_dim=3, capabilities=["encode", "decode", "features", "context_cache",
                                                  "external_initial_noise"])
    simulation = dict(schema_version=2, node_type="simulation", task_id="variable-task",
                      action=deepcopy(action), observation=deepcopy(obs), num_envs=2,
                      control_state_dim=2, task_features_dim=1, reset_mode="indexed",
                      discount={"gamma": .95, "timebase_hz": 30.})
    return inference, simulation


class Inference:
    def __init__(self, meta):
        self.meta = meta
        self.contexts = {}
        self.created = 0
        self.decode_calls = []
        self.encode_calls = []
        self.thread_ids = set()

    def call(self, path, payload=None):
        self.thread_ids.add(threading.get_ident())
        if path == "/health":
            return deepcopy(self.meta)
        if path == "/encode":
            self.encode_calls.append(deepcopy(payload))
            items = []
            for obs in payload["observations"]:
                self.created += 1
                context = f"context-{self.created}"
                self.contexts[context] = obs
                row = dict(context_id=context, features=[obs["state"][0], 2., 3.])
                if payload["include_base_actions"]:
                    row["base_actions"] = np.full((self.meta["action"]["horizon"],
                                                  len(self.meta["action"]["names"])), .2).tolist()
                items.append(row)
            return {"items": items}
        if path == "/decode":
            self.decode_calls.append(deepcopy(payload))
            items = []
            for request in payload["items"]:
                assert request["context_id"] in self.contexts
                noise = np.asarray(request["initial_noise"])
                assert list(noise.shape) == self.meta["noise"]["shape"]
                actions = noise[:, :len(self.meta["action"]["names"])] * .01
                items.append(dict(context_id=request["context_id"], actions=actions.tolist()))
            return {"items": items}
        if path == "/release":
            for context in payload["context_ids"]:
                del self.contexts[context]
            return {"released": len(payload["context_ids"])}
        raise AssertionError(path)


class Simulation:
    def __init__(self, meta):
        self.meta = meta
        self.slots = {}
        self.serial = [0, 0]
        self.resets = []
        self.steps = []
        self.bad_discount = False
        self.lose_step_response = False
        self.delay = 0.

    def observation(self, eid):
        state_dim = len(self.meta["observation"]["state"]["names"])
        step = self.slots[eid]["step_id"]
        marker = self.serial[eid] * 1000 + eid * 100 + step
        return dict(state=[marker / 1000] + [0.] * (state_dim - 1), images={},
                    task="perform this task", model_inputs={}, control_state=[step, .2],
                    task_features=[eid])

    def call(self, path, payload=None):
        if path == "/health":
            return deepcopy(self.meta)
        if path == "/close_run":
            self.closed_run_id = payload["run_id"]
            return {"closed": True}
        if path == "/reset":
            assert payload["env_ids"]
            self.resets.append(deepcopy(payload))
            rows = []
            for eid in payload["env_ids"]:
                if eid in self.slots:
                    assert self.slots[eid]["ended"]
                self.serial[eid] += 1
                self.slots[eid] = dict(env_id=eid, episode_id=f"episode-{eid}-{self.serial[eid]}",
                                       step_id=0, ended=False)
                rows.append({k: v for k, v in self.slots[eid].items() if k != "ended"})
                rows[-1]["observation"] = self.observation(eid)
                rows[-1]["info"] = {}
            return dict(items=rows)
        if path == "/step":
            self.steps.append(deepcopy(payload))
            if self.delay:
                time.sleep(self.delay)
            rows = []
            for item in payload["items"]:
                eid = item["env_id"]
                slot = self.slots[eid]
                assert not slot["ended"]
                assert item["episode_id"] == slot["episode_id"]
                assert item["step_id"] == slot["step_id"]
                slot["step_id"] += 1
                success = eid == 0
                timeout = eid == 1 and slot["step_id"] == 3
                slot["ended"] = success or timeout
                duration = .5 if success else len(item["actions"])
                discount = self.meta["discount"]["gamma"] ** duration
                if self.bad_discount:
                    discount = .1
                rows.append(dict(env_id=eid, episode_id=slot["episode_id"], step_id=slot["step_id"],
                                 observation=self.observation(eid), reward=float(success), discount=discount,
                                 terminated=success, truncated=timeout,
                                 info=dict(success=success, elapsed_control_steps=duration,
                                           physics_steps=int(duration * 4))))
            if self.lose_step_response:
                raise TimeoutError("lost response after physics advanced")
            return dict(items=rows)
        raise AssertionError(path)


def config(output, method="dsrl", steps=8):
    return dict(method=method, device="cpu", output_dir=str(output), seed=5,
                rollout=dict(max_transitions=steps, eval_episodes=4, execution_horizon=2),
                learner=dict(hidden_dim=8, batch_size=2, num_qs=2, num_subset=2),
                update=dict(replay_capacity=32, learning_starts=2, utd=1., max_updates_per_wait=2),
                checkpoint=dict(interval=2, keep=2, include_replay=True))


def make_runner(output, method="dsrl", geometry=(32, 7, 20, 7), steps=8):
    inf, sim = metadata(*geometry)
    inference, simulation = Inference(inf), Simulation(sim)
    cfg = config(output, method, steps)
    if method == "base":
        cfg["mode"] = "eval"
    if method == "action_residual":
        cfg["learner"].update(residual_indices=[0, 2], residual_scale=[.02, .03])
    return GenericTrainer(cfg, inference_client=inference, simulation_client=simulation), inference, simulation


class FakeWandbRun:
    url = "https://wandb.example/test/run"

    def __init__(self):
        self.records, self.definitions, self.finishes = [], [], []
        self.summary = {}

    def define_metric(self, *args, **kwargs):
        self.definitions.append((args, kwargs))

    def log(self, data):
        self.records.append(deepcopy(data))

    def finish(self, **kwargs):
        self.finishes.append(kwargs)


@pytest.fixture
def in_process_wandb(monkeypatch):
    """Exercise upload logic against a fake SDK, never open a real W&B run."""
    def start(monitor):
        monitor.config_path.write_text(json.dumps(monitor.saved_config))

    def close(monitor):
        monitor.stop_path.touch()
        _wandb_worker(str(monitor.output), str(monitor.config_path), monitor.run_id)

    monkeypatch.setattr(trainer_module._WandbMonitor, "start", start)
    monkeypatch.setattr(trainer_module._WandbMonitor, "close", close)


def test_optional_wandb_logs_real_metrics_and_redacts_endpoint_token(tmp_path, monkeypatch, in_process_wandb):
    run, calls = FakeWandbRun(), []

    def initialize(**kwargs):
        calls.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=initialize, Settings=lambda **kw: kw))
    runner, _, _ = make_runner(tmp_path)
    runner.config.wandb.enabled = True
    runner.config.wandb.log_interval_seconds = .001
    runner.config.endpoints.token = "test-private-token"
    runner.config.endpoints.inference = "https://user:private-password@example.test?api_key=private-key"
    result = runner.run()
    assert result["state"] == "completed"
    assert calls[0]["config"]["endpoints"]["token"] is None
    assert "test-private-token" not in json.dumps(calls)
    assert "private-password" not in json.dumps(calls) and "private-key" not in json.dumps(calls)
    assert "inference" not in calls[0]["config"]["endpoints"]
    final = run.records[-1]
    assert final["train/transitions"] == result["transitions"]
    assert final["train/updates"] == result["updates"]
    for key in ("recent_sr", "recent_episodes", "episodes", "successes", "success_rate"):
        assert final[f"rollout/{key}"] == result[key]
    for key in ("critic_loss", "q_target", "q_predicted", "alpha"):
        assert final[f"learner/{key}"] == result["last_update_metrics"][key]
    assert final["performance/transitions_per_second"] == result["transitions_per_second"]
    assert (("learner/*",), {"step_metric": "train/updates", "step_sync": False}) in run.definitions
    assert run.summary["trainer/state"] == "completed"
    assert run.finishes == [{"exit_code": 0}]
    assert json.loads((tmp_path / "wandb_status.json").read_text())["state"] == "finished"


@pytest.mark.parametrize("failure_stage", ["initialization", "logging", "finish"])
def test_wandb_failure_keeps_local_training_and_hides_error_text(
        tmp_path, monkeypatch, capsys, failure_stage, in_process_wandb):
    run = FakeWandbRun()

    def fail(*args, **kwargs):
        raise RuntimeError("test-private-credential-in-sdk-error")

    if failure_stage != "initialization":
        setattr(run, "log" if failure_stage == "logging" else "finish", fail)
    module = SimpleNamespace(init=fail if failure_stage == "initialization" else lambda **kw: run,
                             Settings=lambda **kw: kw)
    monkeypatch.setitem(sys.modules, "wandb", module)
    runner, _, _ = make_runner(tmp_path)
    runner.config.wandb.enabled = True
    result = runner.run()
    assert result["state"] == "completed" and result["updates"] > 0
    assert (tmp_path / "metrics.jsonl").is_file()
    status_text = (tmp_path / "wandb_status.json").read_text()
    status = json.loads(status_text)
    assert status["state"] == "local_fallback" and status["stage"] == failure_stage
    assert "test-private-credential" not in status_text + capsys.readouterr().out


def test_missing_wandb_is_optional_and_training_continues(tmp_path, monkeypatch, in_process_wandb):
    monkeypatch.setitem(sys.modules, "wandb", None)
    runner, _, _ = make_runner(tmp_path / "disabled")
    assert runner.run()["state"] == "completed"
    assert not (runner.output / "wandb_status.json").exists()
    runner, _, _ = make_runner(tmp_path / "enabled")
    runner.config.wandb.enabled = True
    assert runner.run()["state"] == "completed"
    status = json.loads((runner.output / "wandb_status.json").read_text())
    assert status["state"] == "local_fallback" and status["error_type"] == "ModuleNotFoundError"


def test_slow_wandb_startup_never_waits_before_physics(tmp_path, monkeypatch):
    # A real separate process simulates a blocked SDK without importing W&B.
    # Its unbounded exit callback must never become the trainer's callback.
    popen = trainer_module.subprocess.Popen
    commands = []

    def blocked_uploader(command, **kwargs):
        commands.append(command)
        return popen([sys.executable, "-c",
                      "import atexit,time; atexit.register(lambda: time.sleep(60)); time.sleep(60)"], **kwargs)

    monkeypatch.setattr(trainer_module.subprocess, "Popen", blocked_uploader)
    runner, _, sim = make_runner(tmp_path)
    runner.config.wandb.enabled = True
    result = runner.run()
    assert result["state"] == "completed" and sim.steps
    assert "_wandb_worker" in commands[0][2]
    assert runner._wandb.process.poll() is not None
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "completed"
    status = json.loads((tmp_path / "wandb_status.json").read_text())
    assert status["stage"] == "shutdown_timeout" and status["final_remote_metrics_may_be_missing"]


def test_uploader_exits_when_trainer_dies_during_sdk_initialization(tmp_path):
    # A fake SDK in a genuine child process; neither authentication nor a real
    # W&B run is used. The intermediate trainer deliberately skips close().
    sdk = tmp_path / "fake_sdk"
    sdk.mkdir()
    (sdk / "wandb.py").write_text(
        "import time\ndef Settings(**kwargs): return kwargs\n"
        "def init(**kwargs): time.sleep(60)\n")
    environment = os.environ.copy()
    source = str(Path(trainer_module.__file__).resolve().parents[2])
    environment["PYTHONPATH"] = str(sdk) + os.pathsep + source
    code = """from dataclasses import asdict
from pathlib import Path
import sys
from pressb.runtime.trainer import TrainerConfig, _WandbMonitor
output = Path(sys.argv[1])
cfg = TrainerConfig(wandb={"enabled": True})
monitor = _WandbMonitor(output, cfg.wandb, asdict(cfg), "orphan-test")
monitor.start()
(output / "worker.pid").write_text(str(monitor.process.pid))
"""
    trainer_module.subprocess.run([sys.executable, "-c", code, str(tmp_path)], env=environment,
        stdout=trainer_module.subprocess.DEVNULL, stderr=trainer_module.subprocess.DEVNULL,
        check=True, timeout=10)
    pid = int((tmp_path / "worker.pid").read_text())
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline:
            proc_stat = Path(f"/proc/{pid}/stat")
            if not proc_stat.exists() or proc_stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                break
            time.sleep(.1)
        else:
            pytest.fail("Orphaned W&B uploader did not exit")
        status = json.loads((tmp_path / "wandb_status.json").read_text())
        assert status["state"] == "trainer_unavailable" and status["stage"] == "parent_exit"
    finally:
        try:
            os.killpg(pid, trainer_module.signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_wandb_payload_omits_unobserved_sr_and_nonfinite_values():
    payload = _wandb_payload({"transitions": 0, "updates": 0, "recent_sr": None,
                             "success_rate": None, "last_update_metrics": {"alpha": float("nan")}})
    assert payload == {"train/transitions": 0, "train/updates": 0}


@pytest.mark.parametrize("options", [{"log_interval_seconds": 0}, {"init_timeout_seconds": -1},
                                     {"mode": "disabled"}, {"enabled": "yes"}])
def test_invalid_wandb_configuration(options):
    with pytest.raises(ValueError, match="wandb"):
        TrainerConfig(wandb=options)


def test_endpoint_message_limit_must_be_a_positive_integer():
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="max_message_bytes"):
            TrainerConfig(endpoints={"max_message_bytes": value})


@pytest.mark.parametrize("geometry", [(7, 9, 9, 8), (32, 7, 20, 7)])
@pytest.mark.parametrize("method", ["dsrl", "action_residual"])
def test_real_sac_variable_geometry_terminal_observation_and_indexed_reset(tmp_path, method, geometry):
    runner, inf, sim = make_runner(tmp_path, method, geometry)
    result = runner.run()
    assert result["state"] == "completed"
    assert result["transitions"] == 8
    assert result["updates"] > 0
    assert len(runner.replay) == 8
    assert not inf.contexts
    assert len(inf.thread_ids) == 1
    assert threading.get_ident() not in inf.thread_ids
    assert sim.resets[0]["env_ids"] == [0, 1]
    assert sim.resets[1]["env_ids"] == [0]  # Env 1 continues, never group-reset.
    terminal = runner.replay.arrays["terminated"][:8, 0]
    before = runner.replay.arrays["obs"][:8, 0]
    after = runner.replay.arrays["next_obs"][:8, 0]
    assert np.allclose(after[terminal] - before[terminal], .001, atol=1e-6)
    assert all(len(item["actions"]) == 2 for req in sim.steps for item in req["items"])
    if method == "dsrl":
        assert runner.learner.action_dim == geometry[2]
        assert inf.decode_calls
    else:
        assert runner.learner.action_dim == geometry[0] * 2
        for request in sim.steps:
            for item in request["items"]:
                action = np.asarray(item["actions"])
                assert np.allclose(action[:, [i for i in range(geometry[1]) if i not in (0, 2)]], .2)
                assert np.abs(action[:, 0] - .2).max() <= .020001
                assert np.abs(action[:, 2] - .2).max() <= .030001
    assert len(list((tmp_path / "checkpoints").glob("*.pt"))) == 2
    saved = SACLearner_load(result["last_checkpoint"])
    assert saved.updates == result["updates"]  # Cleanup never updates after final checkpoint.


def SACLearner_load(path):
    from pressb.runtime.learner import SACLearner
    return SACLearner.load(path, device="cpu")


def checkpoint_steps(output):
    return [int(path.stem.removeprefix("step_"))
            for path in sorted((output / "checkpoints").glob("step_*.pt"))]


def test_checkpoint_defaults_preserve_every_100k_snapshot():
    cfg = TrainerConfig()
    assert cfg.checkpoint.interval == 100_000
    assert cfg.checkpoint.keep is None


@pytest.mark.parametrize("keep,expected", [(None, [4, 6, 10, 12, 14]), (2, [12, 14])])
def test_checkpoint_absolute_crossings_and_optional_rotation(tmp_path, keep, expected):
    runner, _, _ = make_runner(tmp_path, steps=14)
    runner.config.checkpoint.interval = 3
    runner.config.checkpoint.keep = keep
    result = runner.run()
    # Batches cross 3, 6, 9, 12 at actual steps 4, 6, 10, 12. A relative
    # interval would drift to 4, 8, 12; final 14 must still be preserved.
    assert checkpoint_steps(tmp_path) == expected
    assert result["last_checkpoint"].endswith("step_000000000014.pt")
    for step in expected:
        saved = torch.load(tmp_path / "checkpoints" / f"step_{step:012d}.pt", weights_only=False)
        assert saved["extra"]["transitions"] == step


def test_checkpoint_early_final_resume_keeps_next_absolute_boundary(tmp_path):
    runner, inference, simulation = make_runner(tmp_path, steps=4)
    runner.config.checkpoint.interval = 6
    runner.config.checkpoint.keep = None
    first = runner.run()
    assert checkpoint_steps(tmp_path) == [4]  # An early final is always saved.
    cfg = deepcopy(runner.config)
    cfg.rollout.max_transitions = 14
    cfg.checkpoint.resume = first["last_checkpoint"]
    resumed = GenericTrainer(cfg, inference_client=Inference(inference.meta), simulation_client=Simulation(simulation.meta))
    resumed.run()
    assert checkpoint_steps(tmp_path) == [4, 6, 12, 14]
    assert resumed.transitions == 14


@pytest.mark.parametrize("keep", [0, -1, True, False, 1.5, "2", [], {}])
def test_invalid_checkpoint_keep_rejected(keep):
    with pytest.raises(ValueError, match="checkpoint.keep"):
        TrainerConfig(checkpoint={"keep": keep})


@pytest.mark.parametrize("interval", [None, 0, -1, True, 1.5, "100000"])
def test_invalid_checkpoint_interval_rejected(interval):
    with pytest.raises(ValueError, match="checkpoint.interval"):
        TrainerConfig(checkpoint={"interval": interval})


def frozen_residual_runner(tmp_path, *, fast=False, steps=12):
    noise, inference, simulation = make_runner(tmp_path / "noise", geometry=(4, 3, 3, 3), steps=4)
    noise.config.learner.update(noise_steps=2, noise_scale=.7)
    source = noise.run()["last_checkpoint"]
    runner, inference, simulation = make_runner(tmp_path / "residual", method="action_residual",
                                               geometry=(4, 3, 3, 3), steps=steps)
    runner.config.frozen_dsrl = trainer_module.FrozenDSRLConfig(source, deterministic=True)
    if fast:
        runner.simulation = StateOnlySimulation(simulation.meta)
        runner.config.rollout.terminal_observation_mode = "state_only_if_no_bootstrap"
        runner.config.rollout.fused_step_reset = True
    return noise, runner, inference, runner.simulation


@pytest.mark.parametrize("fast", [False, True])
def test_frozen_dsrl_residual_uses_exact_base_and_never_updates_noise_policy(tmp_path, fast):
    source, runner, inference, simulation = frozen_residual_runner(tmp_path, fast=fast)
    source_state = {key: value.clone() for key, value in source.learner.actor.state_dict().items()}
    result = runner.run()
    assert result["state"] == "completed" and result["transitions"] == 12
    assert runner.learner.updates > 0 and runner.learner.config.method == "action_residual"
    assert runner.frozen_dsrl.config.noise_steps == 2
    assert runner.frozen_dsrl.config.noise_scale == .7
    assert not runner.frozen_dsrl.actor.training
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in runner.frozen_dsrl.actor.parameters())
    assert all(torch.equal(value, runner.frozen_dsrl.actor.state_dict()[key])
               for key, value in source_state.items())
    assert not hasattr(runner.frozen_dsrl, "actor_optimizer")
    assert not inference.contexts
    assert all(request["include_base_actions"] is False for request in inference.encode_calls)
    contexts = [item["context_id"] for request in inference.decode_calls for item in request["items"]]
    assert len(contexts) == inference.created == len(set(contexts))  # No repeated model decoding.
    arrays = {key: value[:len(runner.replay)] for key, value in runner.replay.arrays.items()}
    normalized = source.learner.act(arrays["obs"], deterministic=True)
    expected = np.tile(normalized.reshape(12, 2, 3), (1, 2, 1)) * .7 * .01
    np.testing.assert_allclose(arrays["base_action"].reshape(12, 4, 3), expected, atol=1e-8)
    done = (arrays["terminated"] | arrays["truncated"])[:, 0]
    keep = ~done if fast else np.ones(12, dtype=bool)
    next_normalized = source.learner.act(arrays["next_obs"][keep], deterministic=True)
    expected_next = np.tile(next_normalized.reshape(-1, 2, 3), (1, 2, 1)) * .7 * .01
    np.testing.assert_allclose(arrays["next_base_action"][keep].reshape(-1, 4, 3), expected_next, atol=1e-8)
    if fast:
        assert np.all(arrays["next_base_action"][done] == 0)
    # The commands sent to physics use exactly the replay base, with the learned
    # residual on its selected coordinates; untouched coordinates are identical.
    composed = arrays["base_action"].reshape(12, 4, 3).copy()
    composed[..., [0, 2]] += arrays["action"].reshape(12, 4, 2) * [.02, .03]
    executed = np.asarray([item["actions"] for request in simulation.steps for item in request["items"]])
    np.testing.assert_allclose(executed, composed[:, :2], atol=1e-8)
    assert np.all(arrays["action"][:2] == 0)  # Fresh warmup, not previous DSRL replay.
    init = [json.loads(line) for line in (runner.output / "events.jsonl").read_text().splitlines()
            if json.loads(line)["event"] == "learner_initialized"][0]
    assert (init["updates"], init["transitions"], init["replay_size"], init["resume"]) == (0, 0, 0, None)
    saved = SACLearner_load(result["last_checkpoint"])
    assert saved.checkpoint_extra["frozen_dsrl"]["trainable_parameters"] == 0
    assert saved.checkpoint_extra["frozen_dsrl"]["checkpoint_sha256"] == runner.frozen_dsrl.identity["checkpoint_sha256"]
    verified = json.loads((runner.output / "frozen_dsrl_verification.json").read_text())
    assert verified["matches_initial"] and verified["final"] and verified["transitions"] == 12


@pytest.mark.parametrize("change", ["model", "geometry", "noise_capability", "checksum"])
def test_frozen_dsrl_mismatch_fails_before_physics(tmp_path, change):
    _, runner, inference, simulation = frozen_residual_runner(tmp_path)
    if change == "model":
        inference.meta["model_id"] = "different-weights"
    elif change == "noise_capability":
        inference.meta["capabilities"].remove("external_initial_noise")
    elif change == "checksum":
        runner.config.frozen_dsrl.checkpoint_sha256 = "0" * 64
    else:
        path = Path(runner.config.frozen_dsrl.checkpoint)
        payload = torch.load(path, weights_only=False)
        payload["config"]["obs_dim"] += 1
        torch.save(payload, path)
    with pytest.raises(ValueError, match="fingerprint|geometry|capabilities|SHA256"):
        runner.run()
    assert simulation.resets == [] and inference.encode_calls == []


def test_frozen_dsrl_resume_binds_checkpoint_and_sampling_policy(tmp_path):
    _, runner, inference, simulation = frozen_residual_runner(tmp_path, steps=4)
    result = runner.run()
    cfg = deepcopy(runner.config)
    cfg.output_dir = str(tmp_path / "resume")
    cfg.checkpoint.resume = result["last_checkpoint"]
    cfg.rollout.max_transitions = 8
    resumed = GenericTrainer(cfg, inference_client=Inference(inference.meta), simulation_client=Simulation(simulation.meta))
    assert resumed.run()["transitions"] == 8
    np.testing.assert_array_equal(resumed.replay.arrays["base_action"][:4], runner.replay.arrays["base_action"][:4])
    cfg.output_dir = str(tmp_path / "wrong-sampling")
    cfg.frozen_dsrl.deterministic = False
    wrong = GenericTrainer(cfg, inference_client=Inference(inference.meta), simulation_client=Simulation(simulation.meta))
    with pytest.raises(ValueError, match="fingerprint"):
        wrong.run()
    assert wrong.simulation.resets == []
    cfg.output_dir = str(tmp_path / "wrong-checkpoint")
    cfg.frozen_dsrl.deterministic = True
    path = Path(cfg.frozen_dsrl.checkpoint)
    payload = torch.load(path, weights_only=False)
    payload["actor"]["mean.bias"] += .1
    torch.save(payload, path)
    wrong = GenericTrainer(cfg, inference_client=Inference(inference.meta), simulation_client=Simulation(simulation.meta))
    with pytest.raises(ValueError, match="fingerprint"):
        wrong.run()
    assert wrong.simulation.resets == []


def test_frozen_dsrl_actor_load_preserves_rng(tmp_path):
    source, runner, _, _ = frozen_residual_runner(tmp_path)
    torch.manual_seed(1729)
    before = torch.get_rng_state().clone()
    policy = trainer_module._FrozenDSRLPolicy(runner.config.frozen_dsrl, source.schema_fingerprint,
                                             source.contract, "cpu")
    assert torch.equal(before, torch.get_rng_state())
    observations = np.ones((2, source.learner.config.obs_dim), dtype=np.float32)
    np.testing.assert_array_equal(policy.act(observations), source.learner.act(observations, deterministic=True))
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("settings", [{"checkpoint": ""}, {"checkpoint": "x", "deterministic": "yes"},
                                      {"checkpoint": "x", "checkpoint_sha256": "bad"}])
def test_invalid_frozen_dsrl_configuration(settings):
    with pytest.raises(ValueError, match="frozen_dsrl"):
        TrainerConfig(method="action_residual", frozen_dsrl=settings)
    with pytest.raises(ValueError, match="requires method=action_residual"):
        TrainerConfig(method="dsrl", frozen_dsrl={"checkpoint": "x"})


def test_full_resume_and_model_identity_guard(tmp_path):
    runner, inf, sim = make_runner(tmp_path / "first", steps=4)
    first = runner.run()
    cfg = config(tmp_path / "resumed", steps=8)
    cfg["checkpoint"]["resume"] = first["last_checkpoint"]
    restarted = GenericTrainer(cfg, inference_client=Inference(inf.meta), simulation_client=Simulation(sim.meta))
    result = restarted.run()
    assert result["transitions"] == 8
    assert result["updates"] > first["updates"]
    assert len(restarted.replay) == 8
    assert np.array_equal(restarted.replay.arrays["obs"][:4], runner.replay.arrays["obs"][:4])
    cfg["output_dir"] = str(tmp_path / "mismatched")
    changed = deepcopy(inf.meta)
    changed["model_id"] = "different-weights"
    wrong = GenericTrainer(cfg, inference_client=Inference(changed), simulation_client=Simulation(sim.meta))
    with pytest.raises(ValueError, match="fingerprint"):
        wrong.run()
    assert wrong.simulation.resets == []


def test_failure_does_not_retry_physics_and_releases_contexts(tmp_path):
    runner, inf, sim = make_runner(tmp_path)
    sim.lose_step_response = True
    with pytest.raises(TimeoutError, match="physics advanced"):
        runner.run()
    assert len(sim.steps) == 1
    assert runner.health()["state"] == "failed"
    assert not inf.contexts


def test_discount_must_use_actual_duration(tmp_path):
    runner, inf, sim = make_runner(tmp_path)
    sim.bad_discount = True
    with pytest.raises(ValueError, match="discount disagrees"):
        runner.run()
    assert not inf.contexts


def test_base_evaluation_exact_episode_budget_and_no_learning(tmp_path):
    runner, inf, sim = make_runner(tmp_path, method="base", steps=30)
    runner.config.mode = "eval"
    result = runner.run()
    assert result["state"] == "completed"
    assert result["episodes"] == 4
    assert result["updates"] == 0
    assert not inf.contexts
    assert not (tmp_path / "checkpoints").exists()


def test_updates_overlap_pending_rpc_on_single_learner_thread(tmp_path):
    runner, inf, sim = make_runner(tmp_path, steps=12)
    sim.delay = .03
    runner.config.update.utd = 8
    runner.config.update.max_updates_per_wait = 1
    runner.config.update.max_backlog = 8
    result = runner.run()
    assert result["updates"] > len(sim.steps)
    assert result["dropped_update_credit"] > 0
    assert result["update_backlog"] <= 8


def test_configuration_rejects_ambiguous_shapes_and_unsafe_defaults(tmp_path):
    with pytest.raises(ValueError, match="base supports"):
        TrainerConfig(method="base")
    runner, inf, sim = make_runner(tmp_path)
    runner.config.learner["noise_dim"] = 7
    with pytest.raises(ValueError, match="conflicts"):
        runner.run()
    assert not sim.resets


def test_initial_full_reset_and_all_active_scheduling(tmp_path):
    runner, inf, sim = make_runner(tmp_path, steps=1)
    sim.meta.update(initial_reset="all_envs", scheduling="all_active")
    result = runner.run()
    assert result["transitions"] == 2
    assert result["transition_budget_overshoot"] == 1
    assert sim.resets[0]["env_ids"] == [0, 1]
    assert sim.closed_run_id == runner.run_id


def test_all_active_64_environment_tail_records_true_100k_budget_overshoot(tmp_path):
    runner, _, simulation = state_only_runner(tmp_path, steps=100_000)
    runner.config.rollout.fused_step_reset = True
    runner.config.checkpoint.interval = 100_000
    runner.config.checkpoint.keep = None
    runner.config.update.learning_starts = 100_001
    runner.config.update.utd = 0.
    simulation.meta.update(num_envs=64, initial_reset="all_envs", scheduling="all_active")
    simulation.serial = [0] * 64
    original = runner._negotiate

    def begin_at_last_full_batch():
        original()
        # Model the final full cohort without executing 1,562 preceding CPU
        # batches. The same all-active contract requires all 64 commands even
        # though the remaining requested budget is only 32 transitions.
        runner.transitions = 99_968
        runner._run_initial_transitions = 99_968

    runner._negotiate = begin_at_last_full_batch
    result = runner.run()
    assert len(simulation.steps) == 1 and len(simulation.steps[0]["items"]) == 64
    assert result["transitions"] == 100_032 and result["transition_budget_overshoot"] == 32
    assert simulation.fused_calls[0]["max_active_envs"] == 0
    assert checkpoint_steps(tmp_path) == [100_032]


def test_legacy_full_reset_rejects_requested_subset(tmp_path):
    runner, inf, sim = make_runner(tmp_path)
    sim.meta["initial_reset"] = "all_envs"
    runner.config.rollout.num_envs = 1
    with pytest.raises(ValueError, match="all environments"):
        runner.run()
    assert not sim.resets


def test_training_resume_requires_replay(tmp_path):
    runner, inf, sim = make_runner(tmp_path / "initial", steps=2)
    runner.config.checkpoint.include_replay = False
    first = runner.run()
    cfg = config(tmp_path / "resume")
    cfg["checkpoint"]["resume"] = first["last_checkpoint"]
    runner = GenericTrainer(cfg, inference_client=Inference(inf.meta), simulation_client=Simulation(sim.meta))
    with pytest.raises(ValueError, match="include_replay=true"):
        runner.run()


def test_explicit_clipping_matches_residual_critic_and_preserves_normalized_replay(tmp_path):
    runner, inf, sim = make_runner(tmp_path, method="action_residual", steps=4)
    sim.meta["action"].update(lower=[-.1] * 7, upper=[.1] * 7)
    runner.config.rollout.action_bounds = "clip"
    result = runner.run()
    assert result["projected_action_coordinates"] == 4 * 2 * 7
    assert result["max_projection_delta"] > 0
    assert np.allclose(runner.learner.config.action_upper, [.1] * 7)
    for request in sim.steps:
        for item in request["items"]:
            assert np.allclose(item["actions"], .1)
    batch = runner.learner._prepare_batch(runner.replay.sample(2))
    critic_actions = runner.learner._critic_action(batch["action"], batch["base_action"])
    assert torch.allclose(critic_actions, torch.full_like(critic_actions, .1))
    assert np.abs(runner.replay.arrays["action"][:4]).max() <= 1


def test_out_of_bounds_default_errors_before_physics_and_clip_requires_bounds(tmp_path):
    runner, inf, sim = make_runner(tmp_path / "error", method="base")
    sim.meta["action"].update(lower=[-.1] * 7, upper=[.1] * 7)
    with pytest.raises(ValueError, match="exceeds simulation limits"):
        runner.run()
    assert sim.steps == []
    runner, inf, sim = make_runner(tmp_path / "missing")
    runner.config.rollout.action_bounds = "clip"
    with pytest.raises(ValueError, match="requires simulation"):
        runner.run()
    assert sim.resets == []


def test_eval_rl_requires_trained_checkpoint():
    with pytest.raises(ValueError, match="requires checkpoint"):
        TrainerConfig(method="dsrl", mode="eval")


def test_resume_allows_changed_parallelism_and_eval_starts_fresh_counters(tmp_path):
    runner, inf, sim = make_runner(tmp_path / "initial", steps=4)
    first = runner.run()
    reduced = deepcopy(sim.meta)
    reduced["num_envs"] = 1
    cfg = config(tmp_path / "single_env", steps=6)
    cfg["checkpoint"]["resume"] = first["last_checkpoint"]
    resumed = GenericTrainer(cfg, inference_client=Inference(inf.meta), simulation_client=Simulation(reduced))
    assert resumed.run()["transitions"] == 6
    assert all(req["env_ids"] == [0] for req in resumed.simulation.resets)
    cfg = config(tmp_path / "eval", steps=30)
    cfg["mode"] = "eval"
    cfg["checkpoint"]["resume"] = first["last_checkpoint"]
    evaluation = GenericTrainer(cfg, inference_client=Inference(inf.meta), simulation_client=Simulation(sim.meta))
    result = evaluation.run()
    assert result["episodes"] == 4
    assert result["updates"] == first["updates"]
    assert result["updates_per_second"] == 0.
    assert len(evaluation.replay) == 0


def test_invalid_encoded_features_still_release_all_allocated_contexts(tmp_path):
    runner, inf, sim = make_runner(tmp_path)
    original = inf.call
    def invalid(path, payload=None):
        result = original(path, payload)
        if path == "/encode":
            result["items"][0]["features"] = [1.]
        return result
    inf.call = invalid
    with pytest.raises(ValueError, match="features must be finite with shape"):
        runner.run()
    assert not inf.contexts
    assert not sim.steps


class StateOnlySimulation(Simulation):
    def __init__(self, meta):
        super().__init__(deepcopy(meta))
        self.meta.update(capabilities=["terminal_state_only", "step_and_reset"],
                         terminal_observation_modes=["full", "state_only_if_no_bootstrap"])
        self.terminal_rows = []
        self.continuing_rows = 0
        self.fused_calls = []

    def call(self, path, payload=None):
        if path == "/step_and_reset":
            self.fused_calls.append(deepcopy(payload))
            result = self.call("/step", payload)
            done = [row["env_id"] for row in result["items"] if row["terminated"] or row["truncated"]]
            live = sum(not slot["ended"] for slot in self.slots.values())
            ids = done[:max(0, payload["max_active_envs"] - live)]
            result["reset_items"] = self.call("/reset", {"env_ids": ids,
                "seed": payload["reset_seed"], "options": payload["reset_options"]})["items"] if ids else []
            return result
        result = super().call(path, payload)
        if path == "/step":
            for row in result["items"]:
                if row["terminated"] or row["truncated"]:
                    self.terminal_rows.append(deepcopy(row))
                    if payload.get("terminal_observation_mode") == "state_only_if_no_bootstrap":
                        row["terminal_observation_omitted"] = True
                        row["observation"]["images"] = {}
                else:
                    self.continuing_rows += 1
        return result


def state_only_runner(output, method="dsrl", steps=8):
    runner, inference, sim = make_runner(output, method, steps=steps)
    runner.simulation = StateOnlySimulation(sim.meta)
    runner.config.rollout.terminal_observation_mode = "state_only_if_no_bootstrap"
    return runner, inference, runner.simulation


@pytest.mark.parametrize("method", ["dsrl", "action_residual"])
def test_state_only_mixed_terminal_timeout_replay_and_bellman_target_equivalence(tmp_path, method):
    runner, inference, simulation = state_only_runner(tmp_path, method)
    result = runner.run()
    assert result["state"] == "completed" and result["updates"] > 0
    assert not inference.contexts and not runner.live_contexts
    assert simulation.closed_run_id == runner.run_id
    assert inference.created == sum(len(request["env_ids"]) for request in simulation.resets) + simulation.continuing_rows
    arrays = {name: value[:len(runner.replay)].copy() for name, value in runner.replay.arrays.items()}
    assert arrays["terminated"].any() and arrays["truncated"].any()
    done = (arrays["terminated"] | arrays["truncated"])[:, 0]
    assert np.all(arrays["next_obs"][done, :3] == 0)
    assert np.allclose(arrays["next_obs"][done, 3] - arrays["obs"][done, 3], .001, atol=1e-6)
    assert np.all(arrays["next_obs"][~done, 0] > 0)
    # Restore plausible nonzero terminal image features/base actions. Identical
    # random draws make equality include continuing rows and both done types.
    full = deepcopy(arrays)
    full["next_obs"][done, :3] = [1., 2., 3.]
    if method == "action_residual":
        assert np.all(arrays["next_base_action"][done] == 0)
        full["next_base_action"][done] = .2
    torch.manual_seed(71)
    fast_target = runner.learner._bellman_target(runner.learner._prepare_batch(arrays))
    torch.manual_seed(71)
    full_target = runner.learner._bellman_target(runner.learner._prepare_batch(full))
    assert torch.equal(fast_target, full_target)
    assert torch.equal(fast_target[done], torch.as_tensor(arrays["reward"][done]))
    assert torch.isfinite(fast_target).all()
    checkpoint = SACLearner_load(result["last_checkpoint"])
    assert checkpoint.checkpoint_extra["terminal_observation_mode"] == "state_only_if_no_bootstrap"
    assert np.array_equal(checkpoint.checkpoint_extra["replay"]["arrays"]["next_obs"], arrays["next_obs"])


def test_state_only_all_done_batch_never_encodes_terminal_rows(tmp_path):
    runner, inference, simulation = state_only_runner(tmp_path, steps=4)
    simulation.meta["num_envs"] = 1
    result = runner.run()
    assert result["episodes"] == 4
    assert inference.created == 4  # Exactly initial/reset observations, no terminal encode.
    assert len(inference.decode_calls) == 4
    assert not inference.contexts


@pytest.mark.parametrize("options", [{"learner": {"bootstrap_truncated": True}},
                                      {"learner": {"bootstrap_truncated": 0}},
                                      {"mode": "eval", "method": "base"}])
def test_state_only_requires_explicit_zero_bootstrap_training(options):
    with pytest.raises(ValueError, match="require training with bootstrap_truncated=false"):
        TrainerConfig(rollout={"terminal_observation_mode": "state_only_if_no_bootstrap"}, **options)


def test_state_only_missing_capability_fails_before_reset(tmp_path):
    runner, inference, simulation = make_runner(tmp_path)
    runner.config.rollout.terminal_observation_mode = "state_only_if_no_bootstrap"
    with pytest.raises(ValueError, match="terminal_state_only"):
        runner.run()
    assert not simulation.resets and not inference.contexts


@pytest.mark.parametrize("mode,change", [("state_only_if_no_bootstrap", "missing_done_flag"),
                                       ("state_only_if_no_bootstrap", "omitted_active"),
                                       ("full", "unexpected_omission")])
def test_malformed_terminal_contract_releases_contexts(tmp_path, mode, change):
    runner, inference, simulation = state_only_runner(tmp_path)
    runner.config.rollout.terminal_observation_mode = mode
    original = simulation.call
    def malformed(path, payload=None):
        result = original(path, payload)
        if path == "/step":
            if change == "missing_done_flag":
                result["items"][0].pop("terminal_observation_omitted")
            else:
                index = 1 if change == "omitted_active" else 0
                result["items"][index]["terminal_observation_omitted"] = True
        return result
    simulation.call = malformed
    with pytest.raises(ValueError, match="terminal_observation_omitted"):
        runner.run()
    assert len(simulation.steps) == 1 and not inference.contexts


def test_state_only_resume_preserves_replay_but_rejects_mode_or_bootstrap_change(tmp_path):
    first, inference, simulation = state_only_runner(tmp_path / "first", steps=4)
    checkpoint = first.run()["last_checkpoint"]
    resumed, _, _ = state_only_runner(tmp_path / "resumed", steps=8)
    resumed.config.checkpoint.resume = checkpoint
    assert resumed.run()["transitions"] == 8
    assert np.array_equal(resumed.replay.arrays["next_obs"][:4], first.replay.arrays["next_obs"][:4])
    changed, _, changed_sim = state_only_runner(tmp_path / "changed")
    changed.config.checkpoint.resume = checkpoint
    changed.config.rollout.terminal_observation_mode = "full"
    with pytest.raises(ValueError, match="terminal observation mode differs"):
        changed.run()
    assert not changed_sim.resets
    changed, _, changed_sim = state_only_runner(tmp_path / "bootstrap")
    changed.config.checkpoint.resume = checkpoint
    changed.config.learner["bootstrap_truncated"] = True
    with pytest.raises(ValueError, match="bootstrap_truncated=false"):
        changed.run()
    assert not changed_sim.resets
    # Eval may use full imagery with the same trained model; no replay is loaded.
    evaluation, _, eval_sim = state_only_runner(tmp_path / "eval", steps=30)
    evaluation.config.checkpoint.resume = checkpoint
    evaluation.config.mode = "eval"
    evaluation.config.rollout.terminal_observation_mode = "full"
    assert evaluation.run()["episodes"] == 4
    assert len(evaluation.replay) == 0
    assert all("terminal_observation_mode" not in request for request in eval_sim.steps)


def test_endpoint_image_options_are_typed():
    with pytest.raises(ValueError, match="image_transport"):
        TrainerConfig(endpoints={"image_transport": "raw"})
    with pytest.raises(ValueError, match="preprocess_images"):
        TrainerConfig(endpoints={"preprocess_images": "yes"})
    with pytest.raises(ValueError, match="image_workers"):
        TrainerConfig(endpoints={"image_workers": 0})


def test_fast_mixed_done_combines_reset_and_continuing_in_one_encode(tmp_path):
    runner, inference, simulation = state_only_runner(tmp_path)
    runner.run()
    assert len(inference.encode_calls) == len(simulation.steps) + 1
    assert [len(call["observations"]) for call in inference.encode_calls] == [2, 2, 2, 2, 1]
    first_next = inference.encode_calls[1]["observations"]
    assert first_next[0]["state"][0] == 2.  # Env 0 freshly reset, now episode 2.
    assert first_next[1]["state"][0] == 1.101  # Env 1 continues episode 1.
    assert runner.replay.arrays["next_obs"][0, 3] == pytest.approx(1.001)
    assert runner.replay.arrays["obs"][2, 3] == pytest.approx(2.)
    assert not inference.contexts


def test_fast_no_done_keeps_one_encode_per_step_and_no_extra_reset(tmp_path):
    runner, inference, simulation = state_only_runner(tmp_path)
    original = simulation.call
    def never_ends(path, payload=None):
        result = original(path, payload)
        if path == "/step":
            for row in result["items"]:
                row.update(terminated=False, truncated=False, reward=0.)
                row.pop("terminal_observation_omitted", None)
                row["info"]["success"] = False
                simulation.slots[row["env_id"]]["ended"] = False
        return result
    simulation.call = never_ends
    result = runner.run()
    assert result["episodes"] == 0
    assert len(simulation.resets) == 1
    assert len(inference.encode_calls) == len(simulation.steps) + 1
    assert all(len(call["observations"]) == 2 for call in inference.encode_calls)
    assert not inference.contexts


@pytest.mark.parametrize("failure_stage", ["reset", "encode"])
def test_fast_combined_batch_failure_closes_run_and_releases_all_cache(tmp_path, failure_stage):
    runner, inference, simulation = state_only_runner(tmp_path)
    original_sim, original_inf = simulation.call, inference.call
    def reset_failed(path, payload=None):
        result = original_sim(path, payload)
        if path == "/reset" and len(simulation.resets) > 1:
            raise TimeoutError("lost reset response after state changed")
        return result
    def encode_failed(path, payload=None):
        result = original_inf(path, payload)
        if path == "/encode" and len(inference.encode_calls) > 1:
            # Server allocated contexts for both surviving and reset envs.
            result["items"][0]["features"] = [1.]
        return result
    if failure_stage == "reset":
        simulation.call = reset_failed
        expected = TimeoutError
    else:
        inference.call = encode_failed
        expected = ValueError
    with pytest.raises(expected):
        runner.run()
    assert runner.health()["state"] == "failed"
    assert len(simulation.steps) == 1 and len(simulation.resets) == 2
    assert len(inference.decode_calls) == 1
    assert not inference.contexts and not runner.live_contexts
    assert simulation.closed_run_id == runner.run_id
    if failure_stage == "reset":
        assert "unknown after failed reset/step RPC" in runner.health()["execution_state"]


def test_rpc_timing_logs_successful_scalar_diagnostics_without_payload_or_stale_failures(tmp_path):
    runner, inference, simulation = state_only_runner(tmp_path)
    original = inference.call
    def timed(path, payload=None):
        result = original(path, payload)
        inference.last_call_timing = {"request_bytes": 1024, "exchange_seconds": .2,
                                     "token": "private-token", "invalid_seconds": float("nan")}
        result["timing"] = {"backend_encode_seconds": .1, "images": "private-image"}
        return result
    inference.call = timed
    simulation.lose_step_response = True
    with pytest.raises(TimeoutError):
        runner.run()
    text = (tmp_path / "rpc_timing.jsonl").read_text()
    assert "private-token" not in text and "private-image" not in text and "NaN" not in text
    timings = [json.loads(line) for line in text.splitlines()]
    assert not any(row["path"] == "/step" for row in timings)
    encoded = next(row for row in timings if row["path"] == "/encode")
    assert encoded["client"] == {"request_bytes": 1024, "exchange_seconds": .2}
    assert encoded["server"] == {"backend_encode_seconds": .1}
    assert encoded["node"] == "inference" and encoded["transition"] == 0
    assert encoded["elapsed_seconds"] > 0
    assert any(row["path"] == "/close_run" for row in timings)


@pytest.mark.parametrize("steps,envs", [(8, 2), (5, 2), (4, 1), (1, 2)])
def test_fused_step_reset_has_exact_budget_and_separate_terminal_initial_state(tmp_path, steps, envs):
    runner, inference, simulation = state_only_runner(tmp_path, steps=steps)
    runner.config.rollout.fused_step_reset = True
    simulation.meta["num_envs"] = envs
    result = runner.run()
    assert result["state"] == "completed" and result["transitions"] == steps
    assert len(simulation.fused_calls) == len(simulation.steps)
    assert not inference.contexts
    timings = [json.loads(line) for line in (tmp_path / "rpc_timing.jsonl").read_text().splitlines()]
    assert sum(row["path"] == "/reset" for row in timings) == 1  # Initial reset only.
    assert sum(row["path"] == "/step_and_reset" for row in timings) == len(simulation.steps)
    assert not any(row["path"] == "/step" for row in timings)
    assert simulation.fused_calls[-1]["max_active_envs"] == 0
    assert result["reset_envs"] == sum(len(row["env_ids"]) for row in simulation.resets)
    assert result["reset_calls"] == len(simulation.resets)
    done = runner.replay.arrays["terminated"][:steps, 0] | runner.replay.arrays["truncated"][:steps, 0]
    terminal = runner.replay.arrays["next_obs"][:steps, 3][done]
    assert np.allclose(terminal % 1, np.where((terminal % 1) < .1, .001, .103), atol=1e-6)
    if steps >= 4:
        assert runner.replay.arrays["next_obs"][0, 3] == pytest.approx(1.001)
        assert runner.replay.arrays["obs"][envs, 3] == pytest.approx(2.)


def test_fused_stop_intent_suppresses_reset_at_next_physics_boundary(tmp_path):
    runner, inference, simulation = state_only_runner(tmp_path)
    runner.config.rollout.fused_step_reset = True
    original = inference.call
    def request_stop_after_decode(path, payload=None):
        result = original(path, payload)
        if path == "/decode":
            runner.stop()
        return result
    inference.call = request_stop_after_decode
    result = runner.run()
    assert result["state"] == "stopped" and result["transitions"] == 2
    assert len(simulation.resets) == 1
    assert simulation.fused_calls[0]["max_active_envs"] == 0
    assert result["reset_envs"] == 2 and result["reset_calls"] == 1
    assert not inference.contexts


@pytest.mark.parametrize("change", ["missing_list", "wrong_ids", "stale_episode", "lost_reply"])
def test_fused_bad_or_lost_reply_cleans_contexts_and_never_retries(tmp_path, change):
    runner, inference, simulation = state_only_runner(tmp_path)
    runner.config.rollout.fused_step_reset = True
    original = simulation.call
    def wrong(path, payload=None):
        result = original(path, payload)
        if path == "/step_and_reset":
            if change == "lost_reply":
                raise TimeoutError("lost atomic step/reset response")
            if change == "missing_list":
                result.pop("reset_items")
            elif change == "wrong_ids":
                result["reset_items"] = []
            else:
                result["reset_items"][0]["episode_id"] = result["items"][0]["episode_id"]
        return result
    simulation.call = wrong
    with pytest.raises((ValueError, TimeoutError)):
        runner.run()
    assert len(simulation.fused_calls) == 1 and len(simulation.steps) == 1
    assert not inference.contexts
    assert simulation.closed_run_id == runner.run_id
    assert runner.health()["state"] == "failed"


def test_fused_negotiation_and_resume_guards(tmp_path):
    with pytest.raises(ValueError, match="fused_step_reset"):
        TrainerConfig(rollout={"fused_step_reset": True})
    with pytest.raises(ValueError, match="boolean"):
        TrainerConfig(rollout={"fused_step_reset": 1})
    missing, inference, simulation = state_only_runner(tmp_path / "unsupported")
    missing.config.rollout.fused_step_reset = True
    simulation.meta["capabilities"].remove("step_and_reset")
    with pytest.raises(ValueError, match="step_and_reset"):
        missing.run()
    assert not simulation.resets
    first, _, _ = state_only_runner(tmp_path / "first", steps=4)
    first.config.rollout.fused_step_reset = True
    checkpoint = first.run()["last_checkpoint"]
    resumed, _, _ = state_only_runner(tmp_path / "resumed")
    resumed.config.rollout.fused_step_reset = True
    resumed.config.checkpoint.resume = checkpoint
    assert resumed.run()["transitions"] == 8
    changed, _, changed_sim = state_only_runner(tmp_path / "changed")
    changed.config.checkpoint.resume = checkpoint
    with pytest.raises(ValueError, match="fused_step_reset differs"):
        changed.run()
    assert not changed_sim.resets
