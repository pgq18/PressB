#!/usr/bin/env python3
"""Time a real DPPO trainer from an explicitly selected frozen source tree.

This script does not alter rollout, gradient, likelihood or synchronization
behavior. Checkpoints are skipped only with --skip-checkpoints. Timing logs
live outside the trainer output so its existing empty-directory check remains
effective. No model or environment is replaced by a benchmark implementation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid


PHASES = {
    "_negotiate": "startup", "_reset": "reset", "_collect": "collect",
    "_verify_rollout": "verify_rollout", "_update": "update",
    "_publish": "publish", "_save": "checkpoint",
}


def load_runtime(source_root, expected_sha256=None):
    """Select source before the first PressB import and reject import leakage."""
    root = Path(source_root).expanduser().resolve()
    source = root if (root / "pressb" / "__init__.py").is_file() else root / "src"
    source = source.resolve()
    if not (source / "pressb" / "runtime" / "dppo_trainer.py").is_file():
        raise ValueError(f"No PressB DPPO source found under {root}")
    if any(name == "pressb" or name.startswith("pressb.") for name in sys.modules):
        raise RuntimeError("PressB was imported before --source-root was selected")
    sys.path.insert(0, str(source))
    importlib.invalidate_caches()
    runtime = importlib.import_module("pressb.runtime.dppo_trainer")
    for name, module in tuple(sys.modules.items()):
        if name == "pressb" or name.startswith("pressb."):
            location = getattr(module, "__file__", None)
            if location and not Path(location).resolve().is_relative_to(source):
                raise RuntimeError(f"Unexpected source for {name}: {location}")
    location = Path(runtime.__file__).resolve()
    digest = hashlib.sha256(location.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Trainer SHA256 differs: expected {expected_sha256}, got {digest}")
    for name in ("DPPOTrainer", "DPPOTrainerConfig"):
        if not hasattr(runtime, name):
            raise ValueError(f"Selected source lacks {name}")
    return runtime, dict(source_root=str(source), trainer_module=str(location), trainer_sha256=digest)


class PhaseRecorder:
    """Nested inclusive/exclusive timing around the trainer's bound methods."""

    def __init__(self, phase_log, phase_file, *, skip_checkpoints=False):
        self.phase_log, self.phase_file = Path(phase_log), Path(phase_file)
        self.phase_log.parent.mkdir(parents=True, exist_ok=True)
        self.phase_file.parent.mkdir(parents=True, exist_ok=True)
        self.skip_checkpoints = skip_checkpoints
        self.session_id = uuid.uuid4().hex
        self.runner = None
        self.stack = []
        self.serial = 0
        self.started = time.monotonic()
        self.round_started = None
        self.originals = {}

    def coordinates(self):
        return {key: getattr(self.runner, key, None) for key in
                ("run_id", "transitions", "iterations", "policy_version", "updates", "actor_updates", "critic_updates")}

    def record(self, event, *, active_phase=None, **extra):
        now = time.monotonic()
        record = dict(event=event, session_id=self.session_id, pid=os.getpid(),
                      utc=datetime.now(timezone.utc).isoformat(), unix_seconds=time.time(), monotonic_seconds=now,
                      elapsed_seconds=now - self.started, **self.coordinates(), **extra)
        with self.phase_log.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        phase = active_phase or (self.stack[-1]["phase"] if self.stack else "steady")
        marker = dict(record, phase=phase, active_stack=[dict(phase=f["phase"],
                      phase_id=f["phase_id"], parent_id=f["parent_id"], depth=f["depth"])
                      for f in self.stack])
        temporary = self.phase_file.with_name(self.phase_file.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            temporary.write_text(json.dumps(marker, allow_nan=False) + "\n")
            os.replace(temporary, self.phase_file)
        finally:
            temporary.unlink(missing_ok=True)

    def attach(self, runner):
        self.runner = runner
        for method, phase in PHASES.items():
            original = getattr(runner, method)
            if not callable(original):
                raise ValueError(f"Trainer phase is not callable: {method}")
            self.originals[method] = original
            setattr(runner, method, self._wrap(original, method, phase))

    def _wrap(self, original, method, phase):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            self.serial += 1
            frame = dict(phase=phase, method=method, phase_id=self.serial,
                         parent_id=self.stack[-1]["phase_id"] if self.stack else None,
                         depth=len(self.stack), started=time.monotonic(), child_seconds=0.)
            if method == "_collect":
                self.round_started = frame["started"]
            self.stack.append(frame)
            public = {key: frame[key] for key in ("phase", "method", "phase_id", "parent_id", "depth")}
            self.record("phase_start", **public)
            error = None
            skipped = method == "_save" and self.skip_checkpoints
            try:
                if skipped:
                    self.record("checkpoint_skipped", final=bool(kwargs.get("final", False)), **public)
                    return None
                return original(*args, **kwargs)
            except BaseException as exc:
                error = exc
                raise
            finally:
                duration = time.monotonic() - frame["started"]
                self.stack.pop()
                if self.stack:
                    self.stack[-1]["child_seconds"] += duration
                self.record("phase_end", **public, duration_seconds=duration,
                            exclusive_seconds=max(0., duration - frame["child_seconds"]),
                            child_seconds=frame["child_seconds"], skipped=skipped,
                            outcome="failed" if error else "completed",
                            error_type=type(error).__name__ if error else None,
                            error=str(error) if error else None)
                if method == "_save" and not kwargs.get("final", False):
                    self.record("round_end", duration_seconds=(time.monotonic() - self.round_started)
                                if self.round_started is not None else None,
                                outcome="failed" if error else "completed")
                    self.round_started = None
        return wrapped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Frozen repository root or its src directory")
    parser.add_argument("--expected-trainer-sha256", help="Optionally require an exact frozen trainer file digest")
    parser.add_argument("--config", type=Path, required=True, help="Flat DPPOTrainerConfig JSON")
    parser.add_argument("--output", type=Path, help="Override trainer output_dir")
    parser.add_argument("--phase-log", type=Path, help="JSONL events; default sibling <output>_benchmark/phases.jsonl")
    parser.add_argument("--phase-file", type=Path, help="Atomic current phase marker; default sibling <output>_benchmark/phase.json")
    parser.add_argument("--skip-checkpoints", action="store_true", help="Explicitly skip every _save call, including final save")
    parser.add_argument("--check-only", action="store_true", help="Validate source/config without constructing or running a trainer")
    parser.add_argument("--token-env", default="PRESSB_RL_TOKEN")
    args = parser.parse_args(argv)
    runtime, provenance = load_runtime(args.source_root, args.expected_trainer_sha256)
    data = json.loads(args.config.read_text())
    if args.output is not None:
        data["output_dir"] = str(args.output.expanduser().resolve())
    if os.environ.get(args.token_env):
        data.setdefault("endpoints", {})["token"] = os.environ[args.token_env]
    config = runtime.DPPOTrainerConfig(**data)
    if config.mode != "train":
        raise ValueError("This benchmark requires mode=train")
    output = Path(config.output_dir).expanduser().resolve()
    directory = output.with_name(output.name + "_benchmark")
    phase_log = (args.phase_log or directory / "phases.jsonl").expanduser().resolve()
    phase_file = (args.phase_file or directory / "phase.json").expanduser().resolve()
    if phase_log == phase_file or any(path.is_relative_to(output) for path in (phase_log, phase_file)):
        raise ValueError("Timing files must be distinct and outside the trainer output directory")
    provenance.update(config=str(args.config.resolve()), output_dir=str(output),
                      phase_log=str(phase_log), phase_file=str(phase_file),
                      skip_checkpoints=args.skip_checkpoints,
                      benchmark_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print(json.dumps(dict(event="benchmark_import_verified", **provenance)), flush=True)
    if args.check_only:
        return 0
    recorder = PhaseRecorder(phase_log, phase_file, skip_checkpoints=args.skip_checkpoints)
    recorder.record("benchmark_start", active_phase="startup", provenance=provenance)
    previous_handlers = {}
    result = None
    failure = None
    try:
        runner = runtime.DPPOTrainer(config)
        recorder.attach(runner)
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, lambda *_: runner.stop())
        result = runner.run()
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except BaseException as exc:
        failure = exc
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        state = "failed" if failure else (result.get("state", "completed") if isinstance(result, dict) else "completed")
        recorder.record("benchmark_end", active_phase=state, outcome=state,
                        error_type=type(failure).__name__ if failure else None,
                        error=str(failure) if failure else None, result=result)


if __name__ == "__main__":
    raise SystemExit(main())
