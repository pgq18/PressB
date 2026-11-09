"""CPU instrumentation checks; fixtures are not task success or speed evidence."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from test_rl_nodes_trainer import make_runner


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_dsrl_training.py"
SPEC = importlib.util.spec_from_file_location("benchmark_dsrl_test_module", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def recorder_for(tmp_path, runner, **kwargs):
    recorder = benchmark.PhaseRecorder(tmp_path / "timing/events.jsonl", tmp_path / "timing/phase.json", **kwargs)
    recorder.attach(runner)
    return recorder


def records(recorder):
    return [json.loads(line) for line in recorder.phase_log.read_text().splitlines()]


@pytest.mark.parametrize("skip", [False, True])
def test_real_sac_updates_batch_boundaries_and_checkpoint_opt_in(tmp_path, skip):
    runner, inference, simulation = make_runner(tmp_path / "run", geometry=(4, 2, 2, 2), steps=8)
    recorder = recorder_for(tmp_path, runner, skip_checkpoints=skip)
    try:
        result = runner.run()
        recorder.finish(outcome=result["state"], result=result)
    finally:
        recorder.close()
    rows = records(recorder)
    batches = [row for row in rows if row["event"] == "batch_end"]
    assert result["state"] == "completed" and result["transitions"] == 8
    assert [row["transitions"] for row in batches] == [2, 4, 6, 8]
    assert [row["batch_index"] for row in batches] == [1, 2, 3, 4]
    assert all(row["selected_envs"] == row["transitions_delta"] == 2 for row in batches)
    update_spans = [row for row in rows if row["event"] == "phase_end" and row["kind"] == "update"]
    assert len(update_spans) == result["updates"] > 0
    assert all(row["updates_delta"] == 1 for row in update_spans)
    assert sum(row["updates_delta"] for row in batches) == result["updates"]
    assert batches[-1]["updates"] == result["updates"]
    initial = next(row["identity"] for row in rows if row["event"] == "learner_initial")
    final = next(row["identity"] for row in rows if row["event"] == "learner_final")
    assert initial["updates"] == 0 and final["updates"] == result["updates"]
    assert initial["actor_sha256"] != final["actor_sha256"]
    assert final["optimizers"]["q_optimizer"]["step_max"] == result["updates"]
    assert final["optimizers"]["actor_optimizer"]["state_entries"] > 0
    assert bool(list((tmp_path / "run/checkpoints").glob("*.pt"))) == (not skip)
    assert len([row for row in rows if row["event"] == "checkpoint_skipped"]) == (5 if skip else 0)
    assert len(simulation.steps) == 4 and not inference.contexts
    ends = [row for row in rows if row["event"] == "phase_end"]
    assert all(row["duration_seconds"] >= 0 for row in ends)
    rpc = [row for row in ends if row["kind"] == "rpc"]
    waits = [row for row in ends if row["kind"] == "rpc_wait"]
    assert len(rpc) == len(waits) == sum(result["rpc_calls"].values()) + 2  # final release + close_run after status
    assert {row["thread_id"] for row in rpc}.isdisjoint({row["thread_id"] for row in waits})
    assert all(not stack for stack in recorder.stacks.values())
    marker = json.loads(recorder.phase_file.read_text())
    assert marker["phase"] == "completed" and marker["active_spans"] == []


def test_true_worker_rpc_overlaps_main_update_without_scheduler_replacement(tmp_path):
    runner, _, simulation = make_runner(tmp_path / "run", geometry=(4, 2, 2, 2))
    runner.output.mkdir()
    entered, updated = threading.Event(), threading.Event()
    runner.learner = SimpleNamespace(updates=0)
    runner._can_update = lambda: runner.learner.updates == 0

    def actual_update():
        assert entered.wait(3), "RPC worker was serialized behind update"
        runner.learner.updates += 1
        updated.set()

    def actual_rpc(path, payload):
        assert path == "/step" and payload is payload_identity
        entered.set()
        assert updated.wait(3), "Native asynchronous update was blocked by RPC"
        return {"ok": True}

    payload_identity = {"private": "must-not-be-logged"}
    runner._learn_one = actual_update
    simulation.call = actual_rpc
    recorder = recorder_for(tmp_path, runner)
    original_pending = runner._call_pending.__func__
    with ThreadPoolExecutor(max_workers=1) as executor:
        runner._executor = executor
        assert runner._call(simulation, "/step", payload_identity) == {"ok": True}
    recorder.close()
    assert runner._call_pending.__func__ is original_pending
    assert runner.learner.updates == 1
    ends = [row for row in records(recorder) if row["event"] == "phase_end"]
    rpc = next(row for row in ends if row["kind"] == "rpc")
    update = next(row for row in ends if row["kind"] == "update")
    wait = next(row for row in ends if row["kind"] == "rpc_wait")
    pair = lambda row: (row["start_monotonic_seconds"], row["end_monotonic_seconds"])
    partition = benchmark.interval_partition(*pair(wait), [pair(rpc)], [pair(update)])
    assert partition["overlap_seconds"] > 0
    assert sum(partition[key] for key in ("rpc_only_seconds", "update_only_seconds", "overlap_seconds", "neither_seconds")) == pytest.approx(wait["duration_seconds"])
    assert "must-not-be-logged" not in recorder.phase_log.read_text()


def test_worker_completion_never_waits_for_file_lock_or_writes_files(tmp_path):
    runner, inference, _ = make_runner(tmp_path / "run")
    recorder = recorder_for(tmp_path, runner)
    with recorder.lock, ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(inference.call, "/health").result(timeout=3)
        assert result["node_type"] == "inference"
        assert recorder.phase_log.read_text() == ""
        assert not recorder.phase_file.exists()
    recorder.record("flush")
    recorder.close()
    rows = records(recorder)
    assert [row["event"] for row in rows] == ["phase_start", "phase_end", "flush"]
    assert [row["event_id"] for row in rows] == [1, 2, 3]


def test_update_events_are_buffered_without_per_gradient_file_writes(tmp_path):
    runner, _, _ = make_runner(tmp_path / "run")
    runner.learner = SimpleNamespace(updates=0)

    def update():
        runner.learner.updates += 1
    runner._learn_one = update
    recorder = recorder_for(tmp_path, runner)
    for _ in range(25):
        runner._learn_one()
    assert recorder.phase_log.read_text() == "" and not recorder.phase_file.exists()
    recorder.record("batch_end")
    recorder.close()
    ends = [row for row in records(recorder) if row["event"] == "phase_end"]
    assert len(ends) == 25 and all(row["updates_delta"] == 1 for row in ends)


def test_method_work_timestamps_exclude_start_recording(tmp_path, monkeypatch):
    class Clock:
        value = 100.
        def monotonic(self): return self.value
        def time(self): return self.value + 1000.
    clock = Clock()
    monkeypatch.setattr(benchmark, "time", clock)
    recorder = benchmark.PhaseRecorder(tmp_path / "events.jsonl", tmp_path / "phase.json")
    original_record = recorder.record

    def slow_record(event, **kwargs):
        if event == "phase_start":
            clock.value += 50.
        return original_record(event, **kwargs)

    def workload():
        clock.value += 5.
        return "unchanged"

    recorder.record = slow_record
    assert recorder._invoke(workload, "fixture", "fixture", "method", (), {}) == "unchanged"
    recorder.close()
    row = records(recorder)[-1]
    assert row["duration_seconds"] == 5. and row["pre_call_recording_seconds"] == 50.
    assert row["start_monotonic_seconds"] == 150. and row["end_monotonic_seconds"] == 155.


def test_marker_is_throttled_but_boundaries_are_forced(tmp_path):
    recorder = benchmark.PhaseRecorder(tmp_path / "events.jsonl", tmp_path / "phase.json")
    recorder.marker_interval_seconds = 3600
    recorder.record("first")
    first = recorder.phase_file.read_bytes()
    for _ in range(10):
        recorder.record("phase_notification")
    assert recorder.phase_file.read_bytes() == first
    recorder.record("batch_end")
    assert json.loads(recorder.phase_file.read_text())["event"] == "batch_end"
    recorder.close()
    assert len(records(recorder)) == 12


def test_final_identity_failure_preserves_workload_failure_and_session_end(tmp_path):
    class PoisonedActor:
        def state_dict(self):
            raise RuntimeError("device cannot be read after original failure")
    recorder = benchmark.PhaseRecorder(tmp_path / "events.jsonl", tmp_path / "phase.json")
    recorder.runner = SimpleNamespace(learner=SimpleNamespace(actor=PoisonedActor(), updates=3))
    recorder.finish(outcome="failed", failure=TimeoutError("original workload failure"))
    recorder.close()
    rows = records(recorder)
    assert rows[-2]["identity"] is None and rows[-2]["identity_error_type"] == "RuntimeError"
    assert rows[-1]["event"] == "benchmark_end" and rows[-1]["error_type"] == "TimeoutError"


def test_failed_physics_is_not_retried_and_incomplete_batch_not_counted(tmp_path):
    runner, inference, simulation = make_runner(tmp_path / "run", geometry=(4, 2, 2, 2))
    simulation.lose_step_response = True
    recorder = recorder_for(tmp_path, runner)
    with pytest.raises(TimeoutError, match="lost response") as failure:
        runner.run()
    recorder.finish(outcome="failed", failure=failure.value)
    recorder.close()
    rows = records(recorder)
    assert len(simulation.steps) == 1 and runner.transitions == 0
    assert not any(row["event"] == "batch_end" for row in rows)
    failed = [row for row in rows if row["event"] == "phase_end" and row["outcome"] == "failed"]
    assert {(row["kind"], row.get("path")) for row in failed} == {("rpc", "/step"), ("rpc_wait", "/step")}
    assert rows[-1]["incomplete_batch"] and rows[-1]["outcome"] == "failed"
    assert not inference.contexts and simulation.closed_run_id == runner.run_id


def test_interval_union_clip_intersection_is_disjoint():
    value = benchmark.interval_partition(0., 10., [(-2., 3.), (2., 5.), (8., 14.)], [(1., 2.), (4., 9.), (4.5, 5.)])
    assert value == dict(rpc_only_seconds=4., update_only_seconds=3., overlap_seconds=3.,
                         neither_seconds=0., duration_seconds=10., rpc_union_seconds=7., update_union_seconds=6.)
    assert benchmark.interval_partition(0., 0., [], [(-1., 1.)])["duration_seconds"] == 0


@pytest.mark.parametrize("window,rpc,updates", [((2, 1), [], []), ((0, float("inf")), [], []),
    ((0, 10), [(5, 1)], []), ((0, 10), [], [(0, float("nan"))])])
def test_invalid_intervals_rejected(window, rpc, updates):
    with pytest.raises(ValueError, match="finite and ordered"):
        benchmark.interval_partition(*window, rpc, updates)


def fixture_source(tmp_path):
    source = tmp_path / "snapshot/src"
    package = source / "pressb/runtime"
    package.mkdir(parents=True)
    (source / "pressb/__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    trainer = package / "trainer.py"
    trainer.write_text("class GenericTrainer: pass\nclass TrainerConfig:\n def __init__(self,**data): self.__dict__.update(data)\n")
    learner = package / "learner.py"
    learner.write_text("# source identity fixture only\n")
    return source, trainer, learner


@pytest.mark.parametrize("bad", [None, "trainer", "learner"])
def test_cli_check_only_guards_both_sources_before_constructing_runtime(tmp_path, bad):
    source, trainer, learner = fixture_source(tmp_path)
    config = tmp_path / "config.json"
    output = tmp_path / "output"
    config.write_text(json.dumps(dict(method="dsrl", mode="train", output_dir=str(output))))
    command = [sys.executable, str(SCRIPT), "--source-root", str(source.parent),
               "--config", str(config), "--check-only"]
    for name, path in (("trainer", trainer), ("learner", learner)):
        command += [f"--expected-{name}-sha256", "0" * 64 if bad == name else hashlib.sha256(path.read_bytes()).hexdigest()]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    if bad:
        assert result.returncode != 0 and f"{bad.title()} SHA256 differs" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        receipt = json.loads(result.stdout)
        assert receipt["trainer_module"] == str(trainer)
        assert receipt["learner_module"] == str(learner)
        assert receipt["skip_checkpoints"] is False
    assert not output.exists() and not (tmp_path / "output_benchmark").exists()


def test_preimported_runtime_is_rejected(tmp_path):
    source, _, _ = fixture_source(tmp_path)
    # The real CPU fixture already imported PressB. Selection must fail closed.
    with pytest.raises(RuntimeError, match="before --source-root"):
        benchmark.load_runtime(source)


def test_refuse_overwriting_existing_recording(tmp_path):
    path, marker = tmp_path / "events.jsonl", tmp_path / "phase.json"
    recorder = benchmark.PhaseRecorder(path, marker)
    recorder.record("receipt")
    recorder.close()
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        benchmark.PhaseRecorder(path, marker)
    assert path.read_bytes() == before


@pytest.mark.parametrize("mode", ["normal", "skip", "interrupt", "failure"])
def test_cli_runs_and_finishes_or_stops_through_real_wrapper(tmp_path, mode):
    source, trainer, _ = fixture_source(tmp_path)
    trainer.write_text('''
import os, signal
from pathlib import Path
class TrainerConfig:
    def __init__(self,**data): self.__dict__.update(data)
class Client:
    def call(self,path,payload=None): return {}
class GenericTrainer:
    transitions=episodes=update_credit=dropped_update_credit=_step_requests=0
    learner=None
    def __init__(self,config):
        self.config=config; self.inference=Client(); self.simulation=Client(); self.stopped=False
    def stop(self): self.stopped=True
    def _negotiate(self): pass
    def _reset(self,*args): pass
    def _reset_rows(self,*args): pass
    def _encode(self,*args): pass
    def _actions(self,*args): return []
    def _decode_noise(self,*args): pass
    def _next_slots(self,*args): pass
    def _release(self,*args): pass
    def _call(self,*args): pass
    def _learn_one(self,*args): pass
    def _save(self,*,final=False): Path(self.config.output_dir,'checkpoint_called').touch()
    def _status_write(self,state,*args,**kwargs): pass
    def run(self):
        output=Path(self.config.output_dir)
        assert not output.exists() or not any(output.iterdir())
        output.mkdir(parents=True)
        self._negotiate(); self._reset([0]); self._status_write('running')
        self._actions([{}]); self.transitions+=1
        if self.config.fixture_mode=='failure': raise RuntimeError('offline fixture failure')
        if self.config.fixture_mode=='interrupt': os.kill(os.getpid(),signal.SIGINT)
        self._save(); self._status_write('running'); self._save(final=True)
        return {'state':'stopped' if self.stopped else 'completed','transitions':self.transitions}
''')
    output = tmp_path / "output"
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(method="dsrl", mode="train", output_dir=str(output), fixture_mode=mode)))
    command = [sys.executable, str(SCRIPT), "--source-root", str(source), "--config", str(config)]
    if mode == "skip":
        command.append("--skip-checkpoints")
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert result.returncode == (1 if mode == "failure" else 0), result.stderr
    rows = [json.loads(line) for line in (tmp_path / "output_benchmark/phases.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "benchmark_end"
    assert rows[-1]["outcome"] == {"failure": "failed", "interrupt": "stopped"}.get(mode, "completed")
    assert len([row for row in rows if row["event"] == "batch_end"]) == (0 if mode == "failure" else 1)
    assert (output / "checkpoint_called").exists() == (mode not in ("skip", "failure"))
