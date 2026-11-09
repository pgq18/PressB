"""Offline instrumentation tests; fake fixtures are never benchmark evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_dppo_training.py"
SPEC = importlib.util.spec_from_file_location("benchmark_dppo_training_test_module", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class FakeTrainer:
    transitions = iterations = policy_version = updates = 0

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def _negotiate(self):
        self.calls.append("negotiate")

    def _reset(self, ids, *, initial=False):
        self.calls.append(("reset", ids, initial))

    def _collect(self):
        self._reset([1])
        self.transitions += 64
        return ["actual_records_placeholder_for_test"]

    def _verify_rollout(self, records):
        self.calls.append(("verify", records))

    def _update(self, records):
        self._verify_rollout(records)
        self.updates += 1
        self.iterations += 1
        self._publish()

    def _publish(self):
        if self.fail:
            raise RuntimeError("test publication failure")
        self.policy_version += 1

    def _save(self, *, final=False):
        self.calls.append(("save", final))


@pytest.mark.parametrize("skip", [False, True])
def test_wraps_real_methods_and_checkpoint_opt_in_without_changing_data(tmp_path, skip):
    runner = FakeTrainer()
    recorder = benchmark.PhaseRecorder(tmp_path / "phases.jsonl", tmp_path / "phase.json", skip_checkpoints=skip)
    recorder.attach(runner)
    runner._negotiate()
    records = runner._collect()
    runner._update(records)
    runner._save()
    runner._save(final=True)
    assert runner.transitions == 64 and runner.iterations == runner.policy_version == runner.updates == 1
    assert ("verify", records) in runner.calls
    assert [row for row in runner.calls if row[0] == "save"] == ([] if skip else [("save", False), ("save", True)])
    logs = [json.loads(line) for line in recorder.phase_log.read_text().splitlines()]
    starts = {row["phase_id"]: row for row in logs if row["event"] == "phase_start"}
    ends = [row for row in logs if row["event"] == "phase_end"]
    assert len(ends) == len(starts) == 8
    for row in ends:
        assert row["duration_seconds"] >= row["exclusive_seconds"] >= 0
        assert row["duration_seconds"] == pytest.approx(row["exclusive_seconds"] + row["child_seconds"])
        if row["parent_id"] is not None:
            assert row["depth"] == starts[row["parent_id"]]["depth"] + 1
    publish = next(row for row in ends if row["phase"] == "publish")
    assert starts[publish["parent_id"]]["phase"] == "update"
    reset = next(row for row in ends if row["phase"] == "reset")
    assert starts[reset["parent_id"]]["phase"] == "collect"
    assert len([row for row in logs if row["event"] == "round_end"]) == 1
    assert len([row for row in logs if row["event"] == "checkpoint_skipped"]) == (2 if skip else 0)
    marker = json.loads(recorder.phase_file.read_text())
    assert marker["phase"] == "steady" and marker["active_stack"] == []


def test_nested_failure_preserves_exception_and_clears_stack(tmp_path):
    runner = FakeTrainer(fail=True)
    recorder = benchmark.PhaseRecorder(tmp_path / "phases.jsonl", tmp_path / "phase.json")
    recorder.attach(runner)
    with pytest.raises(RuntimeError, match="publication failure"):
        runner._update([1])
    assert not recorder.stack and runner.policy_version == 0
    failed = [json.loads(line) for line in recorder.phase_log.read_text().splitlines()
              if json.loads(line).get("outcome") == "failed"]
    assert [row["phase"] for row in failed] == ["publish", "update"]


def fixture_source(tmp_path):
    source = tmp_path / "snapshot/src"
    package = source / "pressb/runtime"
    package.mkdir(parents=True)
    (source / "pressb/__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    module = package / "dppo_trainer.py"
    module.write_text('''
import os, signal
from pathlib import Path
class DPPOTrainerConfig:
    def __init__(self, **data): self.__dict__.update(data)
class DPPOTrainer:
    transitions=iterations=policy_version=updates=0
    def __init__(self, config): self.config=config; self.stopped=False
    def stop(self): self.stopped=True
    def _negotiate(self): pass
    def _reset(self,*a,**kw): pass
    def _collect(self):
        self.transitions+=1
        if getattr(self.config,'interrupt',False): os.kill(os.getpid(),signal.SIGINT)
        return [1]
    def _verify_rollout(self,records): pass
    def _update(self,records):
        self._verify_rollout(records); self.iterations+=1; self._publish()
    def _publish(self):
        if getattr(self.config,'fail',False): raise RuntimeError('offline fixture failure')
        self.policy_version+=1
    def _save(self,*,final=False):
        Path(self.config.output_dir,'checkpoint_called').touch()
    def run(self):
        output=Path(self.config.output_dir)
        assert not output.exists() or not any(output.iterdir())
        output.mkdir(parents=True,exist_ok=True)
        self._negotiate(); self._reset([0],initial=True)
        records=self._collect(); self._update(records); self._save(); self._save(final=True)
        return {'state':'stopped' if self.stopped else 'completed','transitions':self.transitions}
''')
    return source, module


@pytest.mark.parametrize("mode", ["normal", "skip", "failure", "interrupt", "check"])
def test_cli_source_selection_completion_failure_and_signal(tmp_path, mode):
    source, module = fixture_source(tmp_path)
    output = tmp_path / "training"
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(mode="train", output_dir=str(output), fail=mode == "failure", interrupt=mode == "interrupt")))
    command = [sys.executable, str(SCRIPT), "--source-root", str(source.parent), "--config", str(config),
               "--expected-trainer-sha256", hashlib.sha256(module.read_bytes()).hexdigest()]
    if mode == "skip": command.append("--skip-checkpoints")
    if mode == "check": command.append("--check-only")
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert result.returncode == (1 if mode == "failure" else 0), result.stderr
    receipt = json.loads(result.stdout.splitlines()[0])
    assert Path(receipt["trainer_module"]) == module and receipt["skip_checkpoints"] == (mode == "skip")
    if mode == "check":
        assert not output.exists() and not (tmp_path / "training_benchmark").exists()
        return
    marker = json.loads((tmp_path / "training_benchmark/phase.json").read_text())
    assert marker["event"] == "benchmark_end"
    assert marker["phase"] == {"failure": "failed", "interrupt": "stopped"}.get(mode, "completed")
    assert (output / "checkpoint_called").exists() == (mode not in ("skip", "failure"))


def test_cli_rejects_source_hash_mismatch_before_runner(tmp_path):
    source, _ = fixture_source(tmp_path)
    result = subprocess.run([sys.executable, str(SCRIPT), "--source-root", str(source),
        "--config", str(tmp_path / "unused.json"), "--expected-trainer-sha256", "0" * 64], capture_output=True, text=True)
    assert result.returncode != 0 and "Trainer SHA256 differs" in result.stderr


def test_rejects_preimported_pressb(monkeypatch, tmp_path):
    source, _ = fixture_source(tmp_path)
    monkeypatch.setitem(sys.modules, "pressb.already_loaded", object())
    with pytest.raises(RuntimeError, match="before --source-root"):
        benchmark.load_runtime(source)
