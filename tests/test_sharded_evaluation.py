"""CPU socket/process tests for the shared-inference evaluation supervisor.

The synthetic plant deliberately makes both trainers enter a simulation step
before either can finish. These are orchestration tests, not physics benchmarks.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_sharded_evaluation.py"


BACKEND_SOURCE = r'''
import json
import os
from pathlib import Path
import time
import numpy as np


def emit(config, event, **values):
    record = json.dumps(dict(event=event, pid=os.getpid(), time=time.monotonic(), **values)) + "\n"
    descriptor = os.open(config["events"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, record.encode())
    finally:
        os.close(descriptor)


def metadata(kind, count=2):
    vector = dict(names=["joint1"], units=["rad"])
    value = dict(schema_version=2, node_type=kind,
        observation=dict(state=vector, cameras=[]),
        action=dict(**vector, horizon=7, control_hz=30,
                    representation="joint_position", frame="robot_base"))
    if kind == "inference":
        value.update(model_id="synthetic-shared-model", frozen=True,
            checkpoint_sha256="a" * 64, weights_sha256="a" * 64,
            source_revision="synthetic-v1", feature_dim=1, noise=dict(shape=[7, 1]),
            capabilities=["encode", "decode", "features", "context_cache", "external_initial_noise"])
    else:
        value.update(task_id="sharded-fixture", num_envs=count, control_state_dim=1,
            task_features_dim=0, reset_mode="indexed", capabilities=["per_env_reset_seeds"],
            discount=dict(gamma=.99, timebase_hz=30), environment_type="simulation",
            source_revision="synthetic-v1", task_config_sha256="b" * 64)
    return value


class Model:
    def __init__(self, config):
        self.config = config
        emit(config, "model_created")
        time.sleep(float(config.get("load_delay", 0)))
        if config.get("wait_for_simulators"):
            deadline = time.monotonic() + 15
            while True:
                trace = [json.loads(line) for line in Path(config["events"]).read_text().splitlines()]
                if len({row["pid"] for row in trace if row["event"] == "simulation_created"}) == 2:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("parallel startup must start simulators while model loads")
                time.sleep(.01)
        emit(config, "model_ready")

    def describe(self):
        return metadata("inference")

    def encode_batch(self, observations, seeds):
        emit(self.config, "encode", workers=sorted({int(obs["state"][0]) for obs in observations}))
        return [(obs["state"], np.asarray([0.])) for obs in observations]

    def decode_batch(self, contexts, noises):
        return [np.zeros((7, 1)) for _ in contexts]

    def close(self):
        emit(self.config, "model_closed")
        if self.config.get("failure") == "model_close":
            raise RuntimeError("synthetic shared-model close failure")


class Plant:
    def __init__(self, config):
        self.config = config
        self.count = config["num_envs"]
        self.ticks = [0] * self.count
        self.steps = 0
        self.resets = 0
        self.events_dir = Path(config["events"]).parent
        health_path = config.get("require_inference_health")
        verified = None
        if health_path:
            health = json.loads(Path(health_path).read_text()) if Path(health_path).is_file() else {}
            verified = health.get("node_type") == "inference" and health.get("ready") is True
        emit(config, "simulation_created", count=self.count, inference_health_verified=verified)
        if health_path and not verified:
            raise RuntimeError("simulator started before inference health was verified")

    def describe(self):
        value = metadata("simulation", self.count)
        value["environment_type"] = self.config.get("reported_environment_type", "simulation")
        return value

    def observation(self, i):
        return dict(state=[float(os.getpid())], control_state=[float(self.ticks[i])],
                    task_features=[], images={}, task="Run the synthetic orchestration fixture.")

    def reset(self, env_ids, seed, options, *, env_seeds=None):
        self.resets += len(env_ids)
        emit(self.config, "reset", env_ids=env_ids, seed=seed, env_seeds=env_seeds)
        for i in env_ids:
            self.ticks[i] = 0
        return [dict(env_id=i, observation=self.observation(i)) for i in env_ids]

    def step(self, commands):
        self.steps += 1
        emit(self.config, "step_started", step=self.steps)
        if self.config.get("barrier", True) and self.steps == 1:
            (self.events_dir / ("entered-" + str(os.getpid()))).touch()
            deadline = time.monotonic() + 15
            while len(list(self.events_dir.glob("entered-*"))) < 2:
                if time.monotonic() > deadline:
                    raise RuntimeError("both trainers must dispatch before either worker finishes")
                time.sleep(.01)
        if self.config.get("failure") == "step":
            emit(self.config, "step_failed")
            raise RuntimeError("lost response after synthetic physics advanced")
        time.sleep(float(self.config.get("step_delay", .03)))
        rows = []
        for command in commands:
            i = command["env_id"]
            self.ticks[i] += 1
            success = i % 2 == 0
            done = success or self.ticks[i] >= 2
            rows.append(dict(env_id=i, observation=self.observation(i), reward=float(success),
                discount=.99 ** len(command["actions"]), terminated=done and success,
                truncated=done and not success,
                info=dict(success=done and success, elapsed_control_steps=len(command["actions"]),
                          physics_steps=4 * len(command["actions"]),
                          terminal_reason="success" if success else "timeout")))
        emit(self.config, "step_finished", step=self.steps)
        return rows

    def status(self):
        return dict(timings=dict(physics_seconds=self.steps * .02, render_seconds=self.steps * .01,
                                step_calls=self.steps, reset_envs=self.resets), ready=True)

    def close_run(self):
        emit(self.config, "close_run")
        if self.config.get("failure") == "close":
            raise RuntimeError("synthetic close_run failure")

    def close(self):
        emit(self.config, "simulation_closed")


def create_model(config):
    return Model(config)


def create_plant(config):
    return Plant(config)
'''


def _read(path):
    return json.loads(Path(path).read_text())


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _ports():
    # Hold all three candidates until the whole consecutive simulator range is
    # known to be free; supervisors still perform their own binding preflight.
    for _ in range(100):
        with socket.socket() as first, socket.socket() as second, socket.socket() as inference:
            first.bind(("127.0.0.1", 0))
            base = first.getsockname()[1]
            if base >= 65535:
                continue
            try:
                second.bind(("127.0.0.1", base + 1))
            except OSError:
                continue
            inference.bind(("127.0.0.1", 0))
            return base, inference.getsockname()[1]
    raise RuntimeError("No free pair of adjacent simulator ports")


@pytest.fixture
def experiment(tmp_path):
    (tmp_path / "sharded_fixture.py").write_text(BACKEND_SOURCE)
    events = tmp_path / "events.jsonl"
    value = dict(schema_version=1, name="sharded-cpu-fixture",
        task=dict(factory="sharded_fixture:create_plant", config=dict(
            num_envs=2, horizon=7, control_hz=30, max_episode_seconds=45,
            terminate_on_success=True, events=str(events))),
        model=dict(factory="sharded_fixture:create_model", config=dict(events=str(events))),
        nodes=dict(inference=dict(server=dict(transport="http"), service=dict(batch_size=2, cache_size=2)),
                   simulation=dict(server=dict(transport="http"))),
        run=dict(method="base", mode="eval", seed=813, endpoints=dict(timeout=20),
                 rollout=dict(max_transitions=100, eval_episodes=5, num_envs=2, execution_horizon=7)))
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(value))
    return path, value, events


def _command(tmp_path, config, *, extra=()):
    simulation, inference = _ports()
    return [sys.executable, str(SCRIPT), "--config", str(config), "--output", str(tmp_path / "result"),
            "--workers", "2", "--num-envs", "2", "--episodes", "5", "--batch-size", "2",
            "--simulation-port", str(simulation), "--inference-port", str(inference),
            "--sim-python", sys.executable, "--model-python", sys.executable,
            "--expected-task-id", "sharded-fixture", "--expected-weights-sha256", "a" * 64,
            "--startup-timeout", "30", "--poll-seconds", ".05", *map(str, extra)]


def _environment(tmp_path):
    return dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]),
                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def _run(tmp_path, command):
    with subprocess.Popen(command, env=_environment(tmp_path), cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            # Allow the supervisor's own signal handler to reap its separate
            # process groups before reporting a failed orchestration test.
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=35)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate(timeout=5)
            pytest.fail("Synthetic supervisor exceeded its deadline:\n" + stdout + stderr)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _diagnostics(tmp_path, completed):
    return completed.stdout + completed.stderr + "\n".join(
        f"{path}:\n{path.read_text()}" for path in (tmp_path / "result").rglob("*.log"))


def _assert_owned_children_gone(status):
    for process in status.get("processes", {}).values():
        if isinstance(process, dict) and "pid" in process:
            with pytest.raises(ProcessLookupError):
                os.kill(process["pid"], 0)


def test_episode_partition_preserves_odd_counts_and_rejects_empty_shards():
    from pressb.runtime.sharded_eval import split_episodes
    assert tuple(split_episodes(5, 2)) == (3, 2)
    assert tuple(split_episodes(100, 3)) == (34, 33, 33)
    for episodes, workers in ((1, 2), (0, 2), (5, 0), (True, 1)):
        with pytest.raises(ValueError):
            split_episodes(episodes, workers)


@pytest.mark.parametrize("event,exception,match", [
    ("signal", InterruptedError, "SIGINT"),
    ("child_exit", RuntimeError, "exited with code 1"),
])
def test_last_health_failure_boundary_blocks_next_launch(tmp_path, monkeypatch, event, exception, match):
    import pressb.runtime.sharded_eval as sharded

    runner = sharded.ShardedEvaluation(SimpleNamespace(output=tmp_path, startup_timeout=1.))
    worker = dict(worker_id=0, config=str(tmp_path / "experiment.json"),
                  experiment=dict(run=dict(endpoints=dict(inference="http://127.0.0.1:12345"))))
    runner.plan = dict(workers=[worker], diagnostic=False)
    child = SimpleNamespace(returncode=None)
    child.poll = lambda: child.returncode
    runner.children["inference"] = child
    runner.state["processes"]["inference"] = dict(log=str(tmp_path / "inference.log"))
    calls = []

    def successful_health(endpoint, path, timeout):
        calls.append(path)
        if event == "signal":
            runner.stopping = True
        else:
            child.returncode = 1
        return dict(schema_version=2, node_type="inference", ready=True)

    monkeypatch.setattr(runner, "call", successful_health)
    monkeypatch.setattr(runner, "validate_health", lambda *_: None)
    with pytest.raises(exception, match=match):
        runner.await_health(inference_only=True)
    assert calls == ["/health"]
    assert _read(tmp_path / "inference.health.json")["ready"] is True

    def unexpected_spawn(*args, **kwargs):
        pytest.fail("A simulator must not start after shutdown or a child exit")

    monkeypatch.setattr(sharded.subprocess, "Popen", unexpected_spawn)
    with pytest.raises(exception, match=match):
        runner.launch("simulation_000", "simulation", sys.executable, worker)
    assert not (tmp_path / "simulation.log").exists()


def test_real_subprocess_workers_share_one_model_and_include_failed_episodes(tmp_path, experiment):
    pytest.importorskip("torch")
    config, original, events = experiment
    completed = _run(tmp_path, _command(tmp_path, config))
    assert completed.returncode == 0, _diagnostics(tmp_path, completed)
    result = _read(tmp_path / "result" / "result.json")
    episodes = _rows(tmp_path / "result" / "episodes.jsonl")
    assert (result["episodes"], result["successes"], result["failures"]) == (5, 3, 2)
    assert result["success_rate"] == 3 / 5
    assert sorted(row["global_episode_id"] for row in episodes) == list(range(5))
    assert len({(row["worker_id"], row["episode_id"]) for row in episodes}) == 5
    assert len({row["worker_id"] for row in episodes}) == 2
    assert sorted(sum(row["worker_id"] == worker for row in episodes)
                  for worker in {row["worker_id"] for row in episodes}) == [2, 3]
    trace = _rows(events)
    assert len([row for row in trace if row["event"] == "model_created"]) == 1
    model_closes = [row for row in trace if row["event"] == "model_closed"]
    assert len(model_closes) == 1
    assert model_closes[0]["time"] > max(row["time"] for row in trace if row["event"] == "close_run")
    simulator_pids = {row["pid"] for row in trace if row["event"] == "simulation_created"}
    assert len(simulator_pids) == 2
    first_finish = min(row["time"] for row in trace if row["event"] == "step_finished")
    assert {row["pid"] for row in trace if row["event"] == "step_started" and row["time"] < first_finish} == simulator_pids
    assert {worker for row in trace if row["event"] == "encode" for worker in row["workers"]} == simulator_pids
    timing = result["timing"]
    assert 0 < timing["evaluation_wall_seconds"] < timing["total_wall_seconds"]
    assert timing["worker_elapsed_seconds_sum"] == pytest.approx(
        sum(worker["runner_status"]["elapsed_seconds"] for worker in result["workers"]))
    step_calls = len([row for row in trace if row["event"] == "step_started"])
    assert timing["rpc_successful_calls"]["simulation/step"]["calls"] == step_calls
    assert timing["simulation_counters_worker_sum"]["step_calls"] == step_calls
    assert timing["simulation_counters_worker_sum"]["physics_seconds"] == pytest.approx(step_calls * .02)
    resets = [row for row in trace if row["event"] == "reset"]
    first_seeds = [next(row["seed"] for row in resets if row["pid"] == pid) for pid in simulator_pids]
    assert len(set(first_seeds)) == 2
    status = _read(tmp_path / "result" / "status.json")
    assert status["status"] == "completed"
    assert status["children_cleaned_up"] is True
    assert len(status["processes"]) == 5
    _assert_owned_children_gone(status)
    assert _read(config) == original


def test_default_startup_verifies_loaded_model_before_starting_simulators(tmp_path, experiment):
    pytest.importorskip("torch")
    config, value, events = experiment
    value["model"]["config"]["load_delay"] = .25
    value["task"]["config"]["require_inference_health"] = str(tmp_path / "result" / "inference.health.json")
    config.write_text(json.dumps(value))
    completed = _run(tmp_path, _command(tmp_path, config))
    assert completed.returncode == 0, _diagnostics(tmp_path, completed)
    trace = _rows(events)
    ready = [row for row in trace if row["event"] == "model_ready"]
    simulators = [row for row in trace if row["event"] == "simulation_created"]
    assert len(ready) == 1 and len(simulators) == 2
    assert all(row["inference_health_verified"] is True for row in simulators)
    assert all(row["time"] > ready[0]["time"] for row in simulators)
    result = _read(tmp_path / "result" / "result.json")
    assert result["episodes"] == 5 and result["startup_order"] == "inference-first"
    assert 0 < result["timing"]["inference_startup_wall_seconds"] <= result["timing"]["startup_wall_seconds"]
    _assert_owned_children_gone(_read(tmp_path / "result" / "status.json"))


def test_explicit_parallel_startup_keeps_model_and_simulator_loading_concurrent(tmp_path, experiment):
    pytest.importorskip("torch")
    config, value, events = experiment
    # The model cannot become ready until both simulator constructors run;
    # this fails if --startup-order parallel accidentally waits for inference.
    value["model"]["config"]["wait_for_simulators"] = True
    config.write_text(json.dumps(value))
    completed = _run(tmp_path, _command(tmp_path, config, extra=("--startup-order", "parallel")))
    assert completed.returncode == 0, _diagnostics(tmp_path, completed)
    trace = _rows(events)
    ready = [row for row in trace if row["event"] == "model_ready"]
    simulators = [row for row in trace if row["event"] == "simulation_created"]
    assert len(ready) == 1 and len(simulators) == 2
    assert all(row["time"] < ready[0]["time"] for row in simulators)
    result = _read(tmp_path / "result" / "result.json")
    assert result["episodes"] == 5 and result["startup_order"] == "parallel"
    assert "inference_startup_wall_seconds" not in result["timing"]
    _assert_owned_children_gone(_read(tmp_path / "result" / "status.json"))


@pytest.mark.parametrize("failure", ["step", "close"])
def test_worker_failure_never_replays_physics_or_publishes_success(tmp_path, experiment, failure):
    pytest.importorskip("torch")
    config, value, events = experiment
    value["task"]["config"].update(failure=failure, barrier=False)
    config.write_text(json.dumps(value))
    completed = _run(tmp_path, _command(tmp_path, config))
    assert completed.returncode != 0, _diagnostics(tmp_path, completed)
    status = _read(tmp_path / "result" / "status.json")
    assert status["status"] == "failed"
    assert status["children_cleaned_up"] is True
    assert not (tmp_path / "result" / "result.json").exists()
    _assert_owned_children_gone(status)
    if failure == "step":
        trace = _rows(events)
        failures = [row["pid"] for row in trace if row["event"] == "step_failed"]
        assert failures
        assert len(failures) == len(set(failures))


def test_shared_model_close_failure_prevents_result_publication(tmp_path, experiment):
    pytest.importorskip("torch")
    config, value, events = experiment
    value["model"]["config"]["failure"] = "model_close"
    config.write_text(json.dumps(value))
    completed = _run(tmp_path, _command(tmp_path, config))
    assert completed.returncode != 0, _diagnostics(tmp_path, completed)
    status = _read(tmp_path / "result" / "status.json")
    assert status["status"] == "failed" and status["children_cleaned_up"]
    assert any("inference" in error for error in status["cleanup_errors"])
    assert not (tmp_path / "result" / "result.json").exists()
    assert len([row for row in _rows(events) if row["event"] == "model_closed"]) == 1
    _assert_owned_children_gone(status)


def test_sigint_stops_owned_children_without_publishing_partial_result(tmp_path, experiment):
    pytest.importorskip("torch")
    config, value, events = experiment
    value["task"]["config"]["step_delay"] = 2
    config.write_text(json.dumps(value))
    command = _command(tmp_path, config)
    with subprocess.Popen(command, env=_environment(tmp_path), cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        try:
            deadline = time.monotonic() + 40
            while not events.exists() or not any(row["event"] == "step_started" for row in _rows(events)):
                assert process.poll() is None, process.communicate(timeout=1)
                assert time.monotonic() < deadline, "Synthetic trainers did not start"
                time.sleep(.01)
            process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode != 0, stdout + stderr
        finally:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=30)
    status = _read(tmp_path / "result" / "status.json")
    assert status["status"] == "stopped"
    assert status["children_cleaned_up"] is True
    assert not (tmp_path / "result" / "result.json").exists()
    _assert_owned_children_gone(status)


@pytest.mark.parametrize("ports", [(65535, 19900), (19000, 19001), (80, 19900)])
def test_invalid_or_overlapping_ports_fail_before_starting_nodes(tmp_path, experiment, ports):
    config, _, events = experiment
    command = _command(tmp_path, config, extra=("--simulation-port", ports[0], "--inference-port", ports[1], "--validate-only"))
    completed = _run(tmp_path, command)
    assert completed.returncode != 0
    assert not events.exists()


def test_validate_only_preserves_task_semantics_and_refuses_reuse(tmp_path, experiment):
    pytest.importorskip("torch")
    config, value, events = experiment
    command = _command(tmp_path, config, extra=("--validate-only",))
    completed = _run(tmp_path, command)
    assert completed.returncode == 0, _diagnostics(tmp_path, completed)
    assert not events.exists()
    configs = [_read(path) for path in (tmp_path / "result" / "workers").rglob("experiment.json")]
    assert len(configs) == 2
    for resolved in configs:
        task = resolved["task"]["config"]
        for key in ("horizon", "control_hz", "max_episode_seconds", "terminate_on_success"):
            assert task[key] == value["task"]["config"][key]
        assert resolved["run"]["endpoints"]["timeout"] == 20
        assert resolved["nodes"]["inference"]["service"]["cache_size"] >= 8
    assert len({item["run"]["seed"] for item in configs}) == 2
    assert len({item["nodes"]["inference"]["server"]["port"] for item in configs}) == 1
    assert len({item["nodes"]["simulation"]["server"]["port"] for item in configs}) == 2
    repeated = _run(tmp_path, command)
    assert repeated.returncode != 0
    assert not events.exists()


def test_real_robot_backend_rejected_without_opening_devices(tmp_path, experiment):
    config, value, events = experiment
    value["task"] = dict(plugin="piper_real", config={})
    config.write_text(json.dumps(value))
    completed = _run(tmp_path, _command(tmp_path, config, extra=("--validate-only",)))
    assert completed.returncode != 0
    assert "physical robot" in (completed.stdout + completed.stderr).lower()
    assert not events.exists()


@pytest.mark.parametrize("mismatch", ["task", "weights", "physical"])
def test_responsive_wrong_service_fails_before_any_reset(tmp_path, experiment, mismatch):
    pytest.importorskip("torch")
    config, value, events = experiment
    extra = ()
    if mismatch == "task":
        extra = ("--expected-task-id", "different-task")
    elif mismatch == "weights":
        extra = ("--expected-weights-sha256", "b" * 64)
    else:
        value["task"]["config"]["reported_environment_type"] = "real_robot"
        config.write_text(json.dumps(value))
    completed = _run(tmp_path, _command(tmp_path, config, extra=extra))
    assert completed.returncode != 0, _diagnostics(tmp_path, completed)
    assert not any(row["event"] == "reset" for row in _rows(events))
    if mismatch == "weights":
        assert not any(row["event"] == "simulation_created" for row in _rows(events))
    status = _read(tmp_path / "result" / "status.json")
    assert status["status"] == "failed" and status["children_cleaned_up"]
    if mismatch == "weights":
        assert set(status["processes"]) == {"inference"}
    _assert_owned_children_gone(status)


def test_occupied_port_is_not_reused_as_an_external_service(tmp_path, experiment):
    pytest.importorskip("torch")
    config, _, events = experiment
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        completed = _run(tmp_path, _command(tmp_path, config,
                         extra=("--inference-port", occupied.getsockname()[1])))
    assert completed.returncode != 0
    assert not events.exists()
    status = _read(tmp_path / "result" / "status.json")
    assert status["status"] == "failed"
    assert status["processes"] == {}


def test_short_throughput_probe_is_never_published_as_a_formal_evaluation(tmp_path, experiment):
    pytest.importorskip("torch")
    config, _, _ = experiment
    completed = _run(tmp_path, _command(tmp_path, config,
                     extra=("--max-transitions-per-worker", "2")))
    assert completed.returncode == 0, _diagnostics(tmp_path, completed)
    result = _read(tmp_path / "result" / "result.json")
    assert result["status"] == "diagnostic_completed"
    assert result["formal_evaluation"] is False
    assert result["requested_episodes"] == 5
    assert result["episodes"] == 2
    assert result["successes"] == 2 and result["failures"] == 0
    assert [row["global_episode_id"] for row in _rows(tmp_path / "result" / "episodes.jsonl")] == [0, 3]
    status = _read(tmp_path / "result" / "status.json")
    assert status["status"] == "diagnostic_completed" and status["children_cleaned_up"]
    _assert_owned_children_gone(status)


def test_underfilled_worker_budget_is_rejected_before_any_node_starts(tmp_path, experiment):
    config, _, events = experiment
    completed = _run(tmp_path, _command(tmp_path, config, extra=("--episodes", "3", "--validate-only")))
    assert completed.returncode != 0
    assert not events.exists()


def test_aggregation_keeps_concurrent_wall_time_separate_from_worker_and_rpc_sums(tmp_path):
    from pressb.runtime.sharded_eval import summarize
    plan = dict(diagnostic=False, requested_episodes=5, checkpoint={}, seed_semantics="fixture", workers=[])
    for index, (count, elapsed) in enumerate(((3, 8.), (2, 11.))):
        directory = tmp_path / str(index)
        run = directory / "run"
        run.mkdir(parents=True)
        rows = [dict(env_id=i % 2, episode_id=f"local-{i}", success=i != 1,
                     info=dict(terminal_reason="timeout" if i == 1 else "success")) for i in range(count)]
        (run / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (run / "status.json").write_text(json.dumps(dict(state="completed", episodes=count,
            successes=count - 1, elapsed_seconds=elapsed, rpc_seconds={"simulation/step": 5. + index})))
        (run / "rpc_timing.jsonl").write_text(json.dumps(dict(node="simulation", path="/step",
            elapsed_seconds=5. + index, client=dict(pack_seconds=.1), server=dict(physics_seconds=2.))) + "\n")
        (directory / "simulation.final_status.json").write_text(json.dumps(dict(fault=None,
            backend=dict(timings=dict(physics_seconds=2., contact_callbacks=7)))))
        plan["workers"].append(dict(worker_id=index, episodes=count, seed=10 + index,
            global_episode_start=0 if index == 0 else 3, run_dir=str(run)))
    summary, rows = summarize(plan, wall_seconds=12., startup_seconds=2.)
    assert (summary["episodes"], summary["successes"], summary["failures"]) == (5, 3, 2)
    assert summary["termination_reasons"] == dict(success=3, timeout=2)
    assert [row["global_episode_id"] for row in rows] == list(range(5))
    timing = summary["timing"]
    assert timing["evaluation_wall_seconds"] == 12.
    assert timing["startup_wall_seconds"] == 2.
    assert timing["worker_elapsed_seconds_sum"] == 19.
    assert timing["rpc_seconds_worker_sum"] == {"simulation/step": 11.}
    assert timing["rpc_successful_calls"]["simulation/step"] == dict(calls=2, elapsed_seconds=11.,
        client=dict(pack_seconds=.2), server=dict(physics_seconds=4.))
    assert timing["simulation_counters_worker_sum"] == dict(physics_seconds=4., contact_callbacks=14)
    # Matching aggregate counts must never hide repeated local episode IDs.
    first = Path(plan["workers"][0]["run_dir"]) / "episodes.jsonl"
    repeated = _rows(first)
    repeated[2] = repeated[0]
    first.write_text("".join(json.dumps(row) + "\n" for row in repeated))
    with pytest.raises(ValueError, match="Repeated"):
        summarize(plan, wall_seconds=12.)
