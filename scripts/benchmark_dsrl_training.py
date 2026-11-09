#!/usr/bin/env python3
"""Observe the real GenericTrainer without changing its asynchronous schedule.

RPC-worker intervals, caller wait envelopes and learner updates are distinct.
Only their interval unions/intersections may be used for wall-time attribution;
nested method durations must not be added. No CUDA synchronization is inserted.
Checkpoints remain enabled unless --skip-checkpoints is explicitly selected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import uuid


METHOD_PHASES = {
    "_negotiate": "startup", "_reset": "reset", "_reset_rows": "reset_rows",
    "_encode": "encode", "_actions": "actions", "_decode_noise": "decode",
    "_next_slots": "next_observation", "_release": "release",
    "_call": "rpc_wait", "_learn_one": "update", "_save": "checkpoint",
}


def verify_imports(source):
    for name, module in tuple(sys.modules.items()):
        if name == "pressb" or name.startswith("pressb."):
            location = getattr(module, "__file__", None)
            if location and not Path(location).resolve().is_relative_to(source):
                raise RuntimeError(f"Unexpected source for {name}: {location}")


def load_runtime(source_root, expected_trainer_sha256=None, expected_learner_sha256=None):
    """Select all PressB imports before importing the trainer or its learner."""
    root = Path(source_root).expanduser().resolve()
    source = root if (root / "pressb/__init__.py").is_file() else root / "src"
    source = source.resolve()
    if not (source / "pressb/runtime/trainer.py").is_file():
        raise ValueError(f"No PressB GenericTrainer source found under {root}")
    if any(name == "pressb" or name.startswith("pressb.") for name in sys.modules):
        raise RuntimeError("PressB was imported before --source-root was selected")
    sys.path.insert(0, str(source))
    importlib.invalidate_caches()
    runtime = importlib.import_module("pressb.runtime.trainer")
    learner = importlib.import_module("pressb.runtime.learner")
    verify_imports(source)
    provenance = {"source_root": str(source)}
    for name, module, expected in (("trainer", runtime, expected_trainer_sha256),
                                   ("learner", learner, expected_learner_sha256)):
        path = Path(module.__file__).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f"{name.title()} SHA256 differs: expected {expected}, got {digest}")
        provenance.update({f"{name}_module": str(path), f"{name}_sha256": digest})
    for name in ("GenericTrainer", "TrainerConfig"):
        if not hasattr(runtime, name):
            raise ValueError(f"Selected source lacks {name}")
    return runtime, provenance


def learning_identity(learner):
    """Small begin/end receipt; never called per update or inside a batch."""
    if learner is None:
        return None
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(learner.actor.state_dict().items()):
        value = tensor.detach().contiguous().cpu()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    optimizers = {}
    for name in ("actor_optimizer", "q_optimizer", "alpha_optimizer"):
        optimizer = getattr(learner, name)
        steps = [float(state["step"].item() if hasattr(state["step"], "item") else state["step"])
                 for state in optimizer.state.values() if "step" in state]
        optimizers[name] = dict(state_entries=len(optimizer.state), step_entries=len(steps),
                                step_min=min(steps) if steps else None,
                                step_max=max(steps) if steps else None)
    return dict(actor_sha256=digest.hexdigest(), updates=int(learner.updates), optimizers=optimizers)


def interval_partition(start, end, rpc_intervals, update_intervals):
    """Disjoint wall-time partition, clipping and unioning overlapping intervals.

    Caller chooses actual worker RPC intervals or caller wait envelopes. The
    former is observed host concurrency, not a claim of concurrent CUDA kernels.
    """
    if not all(math.isfinite(value) for value in (start, end)) or end < start:
        raise ValueError("Window endpoints must be finite and ordered")
    events = [(start, 0, 0), (end, 0, 0)]
    for kind, intervals in enumerate((rpc_intervals, update_intervals)):
        for left, right in intervals:
            if not all(math.isfinite(value) for value in (left, right)) or right < left:
                raise ValueError("Interval endpoints must be finite and ordered")
            left, right = max(start, left), min(end, right)
            if right <= left:
                continue
            events.extend([(left, int(kind == 0), int(kind == 1)),
                           (right, -int(kind == 0), -int(kind == 1))])
    totals = dict(rpc_only_seconds=0., update_only_seconds=0., overlap_seconds=0., neither_seconds=0.)
    active_rpc = active_update = 0
    previous = start
    for point, rpc_delta, update_delta in sorted(events):
        key = ("overlap_seconds" if active_rpc and active_update else
               "rpc_only_seconds" if active_rpc else
               "update_only_seconds" if active_update else "neither_seconds")
        totals[key] += point - previous
        active_rpc += rpc_delta
        active_update += update_delta
        previous = point
    totals.update(duration_seconds=end - start,
                  rpc_union_seconds=totals["rpc_only_seconds"] + totals["overlap_seconds"],
                  update_union_seconds=totals["update_only_seconds"] + totals["overlap_seconds"])
    return totals


class PhaseRecorder:
    """Thread-local nesting with a shared event stream; no payloads are logged."""

    def __init__(self, phase_log, phase_file, *, skip_checkpoints=False):
        self.phase_log, self.phase_file = Path(phase_log), Path(phase_file)
        if self.phase_log == self.phase_file:
            raise ValueError("Phase log and marker must be distinct")
        if self.phase_file.exists():
            raise FileExistsError(self.phase_file)
        for path in (self.phase_log, self.phase_file):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.phase_log.open("x", buffering=1)
        self.skip_checkpoints = skip_checkpoints
        self.session_id = uuid.uuid4().hex
        self.started = time.monotonic()
        self.runner = None
        self.stacks = {}
        self.lock = threading.RLock()
        self.serial = self.event_serial = self.batch_count = 0
        self.batch = None
        self.originals = {}
        self.recording_seconds = 0.
        self.pending_rpc_events = queue.SimpleQueue()
        self.last_marker_at = 0.
        self.marker_interval_seconds = .1

    def _write_row(self, row):
        self.event_serial += 1
        row["event_id"] = self.event_serial
        row["utc"] = datetime.fromtimestamp(row["unix_seconds"], timezone.utc).isoformat()
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")

    def _flush_rpc_events(self):
        while True:
            try:
                self._write_row(self.pending_rpc_events.get_nowait())
            except queue.Empty:
                return

    def _queue_rpc_event(self, event, *, unix_seconds, monotonic_seconds, **extra):
        # No disk, formatting or shared file lock in the I/O worker: keeping a
        # completed future pending for logging could change native SAC updates.
        self.pending_rpc_events.put(dict(event=event, session_id=self.session_id,
            pid=os.getpid(), thread_id=threading.get_ident(), unix_seconds=unix_seconds,
            monotonic_seconds=monotonic_seconds, elapsed_seconds=monotonic_seconds - self.started,
            **self.coordinates(), **extra))

    def coordinates(self):
        runner = self.runner
        learner = getattr(runner, "learner", None)
        return dict(run_id=getattr(runner, "run_id", None),
                    transitions=getattr(runner, "transitions", 0),
                    updates=getattr(learner, "updates", 0),
                    episodes=getattr(runner, "episodes", 0),
                    update_credit=getattr(runner, "update_credit", 0.),
                    dropped_update_credit=getattr(runner, "dropped_update_credit", 0.),
                    step_requests=getattr(runner, "_step_requests", 0), batches=self.batch_count)

    def record(self, event, *, active_phase=None, **extra):
        started = time.monotonic()
        unix = time.time()
        with self.lock:
            self._flush_rpc_events()
            row = dict(event=event, session_id=self.session_id,
                       pid=os.getpid(), thread_id=threading.get_ident(),
                       utc=datetime.fromtimestamp(unix, timezone.utc).isoformat(),
                       unix_seconds=unix, monotonic_seconds=started,
                       elapsed_seconds=started - self.started,
                       recording_seconds=self.recording_seconds, **self.coordinates(), **extra)
            self._write_row(row)
            force = active_phase is not None or event in ("batch_start", "batch_end")
            if force or started - self.last_marker_at >= self.marker_interval_seconds:
                active = [{key: frame[key] for key in
                           ("phase_id", "phase", "method", "kind", "parent_id", "thread_id", "depth")}
                          for stack in self.stacks.values() for frame in stack]
                phases = sorted({frame["phase"] for frame in active})
                marker = dict(row, phase=active_phase or ("+".join(phases) if phases else "between_calls"),
                              active_spans=active,
                              active_spans_scope="caller-thread snapshot; RPC/update events queued; marker throttled 100ms")
                temporary = self.phase_file.with_name(self.phase_file.name + "." + self.session_id + ".tmp")
                try:
                    temporary.write_text(json.dumps(marker, allow_nan=False) + "\n")
                    os.replace(temporary, self.phase_file)
                finally:
                    temporary.unlink(missing_ok=True)
                self.last_marker_at = started
            self.recording_seconds += time.monotonic() - started
        return row

    def _invoke(self, original, method, phase, kind, args, kwargs, extra=None, skip=False):
        tid = threading.get_ident()
        with self.lock:
            stack = self.stacks.setdefault(tid, [])
            self.serial += 1
            frame = dict(phase_id=self.serial, phase=phase, method=method, kind=kind,
                         parent_id=stack[-1]["phase_id"] if stack else None,
                         thread_id=tid, depth=len(stack), envelope_start_monotonic_seconds=time.monotonic(),
                         envelope_start_unix_seconds=time.time())
            public = {key: value for key, value in frame.items() if key != "thread_id"}
            public.update(extra or {})
            stack.append(frame)
        if kind == "update":
            self._queue_rpc_event("phase_start", unix_seconds=time.time(),
                                  monotonic_seconds=time.monotonic(), **public)
        else:
            self.record("phase_start", **public)
        before = self.coordinates()
        failure = None
        start_mono, start_unix = time.monotonic(), time.time()
        try:
            if skip:
                self.record("checkpoint_skipped", final=bool(kwargs.get("final", False)))
                return None
            return original(*args, **kwargs)
        except BaseException as exc:
            failure = exc
            raise
        finally:
            end_mono, end_unix = time.monotonic(), time.time()
            with self.lock:
                assert self.stacks[tid].pop() is frame
            after = self.coordinates()
            end_fields = dict(**public, start_monotonic_seconds=start_mono, start_unix_seconds=start_unix,
                              end_monotonic_seconds=end_mono, end_unix_seconds=end_unix,
                              duration_seconds=end_mono - start_mono,
                              pre_call_recording_seconds=start_mono - frame["envelope_start_monotonic_seconds"],
                              updates_delta=after["updates"] - before["updates"], skipped=skip,
                              outcome="failed" if failure else "completed",
                              error_type=type(failure).__name__ if failure else None)
            if kind == "update":
                self._queue_rpc_event("phase_end", unix_seconds=end_unix,
                                      monotonic_seconds=end_mono, **end_fields)
            else:
                self.record("phase_end", **end_fields)

    def _method_wrapper(self, original, method, phase):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if method == "_actions":
                if self.batch is not None:
                    raise RuntimeError("New actions before prior batch reached running status")
                self.batch = self.record("batch_start", selected_envs=len(args[0]))
            extra = {}
            if method == "_call":
                extra = dict(node="inference" if args[0] is self.runner.inference else "simulation", path=args[1])
            result = self._invoke(original, method, phase,
                                  "rpc_wait" if method == "_call" else
                                  "update" if method == "_learn_one" else "method",
                                  args, kwargs, extra, method == "_save" and self.skip_checkpoints)
            if method == "_negotiate":
                self.record("learner_initial", identity=learning_identity(self.runner.learner))
            return result
        return wrapped

    def attach(self, runner):
        if self.runner is not None:
            raise RuntimeError("Recorder already attached")
        # Validate before mutation; the supported GenericTrainer API is explicit.
        for method in (*METHOD_PHASES, "_status_write"):
            if not callable(getattr(runner, method, None)):
                raise ValueError(f"Trainer phase is not callable: {method}")
        for client in (runner.inference, runner.simulation):
            if not callable(getattr(client, "call", None)):
                raise ValueError("Node client call is not callable")
        if runner.inference is runner.simulation:
            raise ValueError("Inference and simulation must have distinct client identities")
        self.runner = runner
        for method, phase in METHOD_PHASES.items():
            self.originals[method] = getattr(runner, method)
            setattr(runner, method, self._method_wrapper(self.originals[method], method, phase))
        original_status = runner._status_write
        self.originals["_status_write"] = original_status

        @functools.wraps(original_status)
        def status(state, *args, **kwargs):
            result = original_status(state, *args, **kwargs)
            if state == "running" and self.batch is not None:
                before, self.batch = self.batch, None
                self.batch_count += 1
                now = time.monotonic()
                self.record("batch_end", batch_index=self.batch_count,
                            start_monotonic_seconds=before["monotonic_seconds"],
                            start_unix_seconds=before["unix_seconds"],
                            duration_seconds=now - before["monotonic_seconds"],
                            transitions_delta=runner.transitions - before["transitions"],
                            updates_delta=self.coordinates()["updates"] - before["updates"],
                            selected_envs=before["selected_envs"], outcome="completed")
            return result
        runner._status_write = status
        for node in ("inference", "simulation"):
            client = getattr(runner, node)
            original = client.call
            self.originals[node + ".call"] = original

            def make_rpc(call, node_name):
                @functools.wraps(call)
                def rpc(path, payload=None):
                    phase = ("step" if path in ("/step", "/step_and_reset") else path.lstrip("/"))
                    begin_mono, begin_unix = time.monotonic(), time.time()
                    public = dict(phase_id="rpc-" + uuid.uuid4().hex, phase=phase, method="client.call",
                                  kind="rpc", parent_id=None, depth=0, node=node_name, path=path,
                                  start_monotonic_seconds=begin_mono, start_unix_seconds=begin_unix)
                    self._queue_rpc_event("phase_start", unix_seconds=begin_unix,
                                          monotonic_seconds=begin_mono, **public)
                    failure = None
                    try:
                        return call(path, payload)
                    except BaseException as exc:
                        failure = exc
                        raise
                    finally:
                        end_mono, end_unix = time.monotonic(), time.time()
                        self._queue_rpc_event("phase_end", unix_seconds=end_unix,
                            monotonic_seconds=end_mono, end_monotonic_seconds=end_mono,
                            end_unix_seconds=end_unix, duration_seconds=end_mono - begin_mono,
                            outcome="failed" if failure else "completed",
                            error_type=type(failure).__name__ if failure else None, **public)
                return rpc
            client.call = make_rpc(original, node)

    def finish(self, *, outcome, result=None, failure=None):
        identity, identity_error = None, None
        try:
            identity = learning_identity(getattr(self.runner, "learner", None))
        except Exception as error:
            # A poisoned device after failed update must not hide that failure.
            identity_error = type(error).__name__
        self.record("learner_final", identity=identity, identity_error_type=identity_error)
        self.record("benchmark_end", active_phase=outcome, outcome=outcome, result=result,
                    incomplete_batch=self.batch is not None,
                    error_type=type(failure).__name__ if failure else None)

    def close(self):
        self._flush_rpc_events()
        self.stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-trainer-sha256")
    parser.add_argument("--expected-learner-sha256")
    parser.add_argument("--config", type=Path, required=True, help="Flat TrainerConfig JSON; method=dsrl, mode=train")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase-log", type=Path)
    parser.add_argument("--phase-file", type=Path)
    parser.add_argument("--skip-checkpoints", action="store_true", help="Skip all _save calls, including final save")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--token-env", default="PRESSB_RL_TOKEN")
    args = parser.parse_args(argv)
    runtime, provenance = load_runtime(args.source_root, args.expected_trainer_sha256,
                                       args.expected_learner_sha256)
    data = json.loads(args.config.read_text())
    if args.output is not None:
        data["output_dir"] = str(args.output.expanduser().resolve())
    if os.environ.get(args.token_env):
        data.setdefault("endpoints", {})["token"] = os.environ[args.token_env]
    config = runtime.TrainerConfig(**data)
    if config.mode != "train" or config.method != "dsrl":
        raise ValueError("This benchmark requires method=dsrl and mode=train")
    output = Path(config.output_dir).expanduser().resolve()
    directory = output.with_name(output.name + "_benchmark")
    phase_log = (args.phase_log or directory / "phases.jsonl").expanduser().resolve()
    phase_file = (args.phase_file or directory / "phase.json").expanduser().resolve()
    if phase_log == phase_file or any(path.is_relative_to(output) for path in (phase_log, phase_file)):
        raise ValueError("Timing files must be distinct and outside the trainer output directory")
    provenance.update(config=str(args.config.resolve()),
                      config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                      output_dir=str(output), phase_log=str(phase_log), phase_file=str(phase_file),
                      skip_checkpoints=args.skip_checkpoints,
                      benchmark_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print(json.dumps(dict(event="benchmark_import_verified", **provenance)), flush=True)
    if args.check_only:
        return 0
    recorder = PhaseRecorder(phase_log, phase_file, skip_checkpoints=args.skip_checkpoints)
    recorder.record("benchmark_start", active_phase="startup", provenance=provenance)
    previous_handlers, result, failure = {}, None, None
    try:
        runner = runtime.GenericTrainer(config)
        recorder.attach(runner)
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, lambda *_: runner.stop())
        result = runner.run()
        verify_imports(Path(provenance["source_root"]))
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except BaseException as exc:
        failure = exc
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        state = "failed" if failure else (result.get("state", "completed") if isinstance(result, dict) else "completed")
        try:
            recorder.finish(outcome=state, result=result, failure=failure)
        finally:
            recorder.close()


if __name__ == "__main__":
    raise SystemExit(main())
