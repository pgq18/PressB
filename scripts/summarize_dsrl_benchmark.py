#!/usr/bin/env python3
"""Audit and summarize a completed, instrumented 64-environment DSRL run.

This offline tool uses only the standard library. RPC worker spans and learner
spans may overlap: their inclusive durations are never added as wall time.
The source JSONL order is immaterial; completed spans carry their own clocks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import posixpath
import sys


WRAPPER = Path(__file__).with_name("benchmark_dsrl_training.py")
_SPEC = importlib.util.spec_from_file_location("_dsrl_timing_partition", WRAPPER)
_WRAPPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_WRAPPER)
interval_partition = _WRAPPER.interval_partition


class AuditError(ValueError):
    """The logs cannot substantiate a completed benchmark."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def close(actual, expected, message, *, abs_tol=1e-6):
    require(math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=abs_tol),
            f"{message}: actual={actual}, expected={expected}")


def read_jsonl(path, *, optional=False):
    if optional and not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                value = json.loads(line)
            except ValueError as error:
                raise AuditError(f"Invalid JSON at {path}:{number}") from error
            require(isinstance(value, dict), f"Expected object at {path}:{number}")
            rows.append(value)
    return rows


def unique_event(rows, event):
    selected = [row for row in rows if row.get("event") == event]
    require(len(selected) == 1, f"Expected exactly one {event}, found {len(selected)}")
    return selected[0]


def span_bounds(row):
    return row["start_monotonic_seconds"], row["end_monotonic_seconds"]


def clipped(row, start, end):
    left, right = span_bounds(row)
    return max(0., min(end, right) - max(start, left))


def validate_spans(rows):
    starts, ends = {}, {}
    for row in rows:
        event = row.get("event")
        if event not in ("phase_start", "phase_end"):
            continue
        key = row["phase_id"]
        selected = starts if event == "phase_start" else ends
        require(key not in selected, f"Duplicate {event} for phase {key}")
        selected[key] = row
    require(set(starts) == set(ends), "Missing phase start or final phase end")
    require(bool(ends), "No completed operation spans")
    for key, row in ends.items():
        require(row.get("outcome") == "completed", f"Failed phase {key}")
        for field in ("kind", "method", "phase"):
            require(starts[key].get(field) == row.get(field), f"Phase {key} identity changed: {field}")
        left, right = span_bounds(row)
        require(math.isfinite(left) and math.isfinite(right) and right >= left,
                f"Invalid phase {key} work interval")
        close(row["duration_seconds"], right - left, f"Phase {key} duration")
    # The native runner has one IO worker and one serial learner. Overlap across
    # these two kinds is expected, overlap within either kind is not.
    for kind in ("rpc", "update"):
        ordered = sorted((row for row in ends.values() if row["kind"] == kind), key=span_bounds)
        for previous, following in zip(ordered, ordered[1:]):
            require(previous["end_monotonic_seconds"] <= following["start_monotonic_seconds"] + 1e-9,
                    f"Overlapping {kind} spans contradict the native serial worker")
    return sorted(ends.values(), key=span_bounds)


def timing_window(start, end, rpc, updates, waits):
    pairs = lambda selected: [span_bounds(row) for row in selected]
    partition = interval_partition(start, end, pairs(rpc), pairs(updates))
    routes = {}
    grouped = defaultdict(list)
    for row in rpc:
        if clipped(row, start, end) > 0:
            grouped[f"{row['node']}:{row['path']}"].append(row)
    for route, selected in sorted(grouped.items()):
        part = interval_partition(start, end, pairs(selected), pairs(updates))
        seconds = sum(clipped(row, start, end) for row in selected)
        routes[route] = dict(count=len(selected), total_seconds=seconds,
            mean_seconds=seconds / len(selected), overlap_seconds=part["overlap_seconds"],
            rpc_only_seconds=part["rpc_only_seconds"],
            partial_calls=sum(span_bounds(row)[0] < start or span_bounds(row)[1] > end for row in selected))
    # Caller wait envelopes may contain learner work after the worker finished.
    # They are diagnostic inclusive costs, never a fifth wall partition slice.
    wait_seconds = sum(clipped(row, start, end) for row in waits)
    caller_waits = dict(count=sum(clipped(row, start, end) > 0 for row in waits),
                        inclusive_seconds=wait_seconds,
                        semantics="Caller _call envelope; may outlast worker RPC and overlap learning; do not add to partition")
    close(sum(row["total_seconds"] for row in routes.values()), partition["rpc_union_seconds"],
          "RPC route seconds versus worker union")
    close(sum(row["overlap_seconds"] for row in routes.values()), partition["overlap_seconds"],
          "RPC route overlap versus global overlap")
    return partition, routes, caller_waits


def optimizer_steps(identity, name, expected):
    value = identity["optimizers"][name]
    if expected:
        require(value["state_entries"] > 0 and value["step_entries"] > 0,
                f"{name} has no real optimizer state")
        close(value["step_min"], expected, f"{name} minimum step")
        close(value["step_max"], expected, f"{name} maximum step")
    else:
        require(value["step_entries"] == 0 or value["step_min"] == value["step_max"] == 0,
                f"Unexpected {name} optimizer work")
    return value


def summarize(training_dir, phases_path, start_transition, expected_transitions):
    training_dir, phases_path = Path(training_dir).resolve(), Path(phases_path).resolve()
    paths = {name: training_dir / name for name in
             ("config.json", "status.json", "metrics.jsonl", "episodes.jsonl")}
    config = json.loads(paths["config.json"].read_text())
    status = json.loads(paths["status.json"].read_text())
    metrics = read_jsonl(paths["metrics.jsonl"])
    episodes = read_jsonl(paths["episodes.jsonl"], optional=True)
    phases = read_jsonl(phases_path)
    require(config.get("method") == "dsrl" and config.get("mode") == "train", "Expected DSRL training config")
    require(not config.get("checkpoint", {}).get("resume"), "Fresh benchmark must not resume")
    require(config["rollout"]["num_envs"] == 64, "Expected 64 parallel environments")
    require(config["rollout"]["max_transitions"] == expected_transitions, "Config transition budget differs")
    require(0 < start_transition < expected_transitions and start_transition % 64 == 0
            and expected_transitions % 64 == 0, "Expected positive complete 64-transition boundaries")
    require(status.get("state") == "completed" and status["transitions"] == expected_transitions,
            "Training did not complete the exact budget")
    benchmark_start = unique_event(phases, "benchmark_start")
    benchmark_end = unique_event(phases, "benchmark_end")
    run_id = status.get("run_id")
    require(isinstance(run_id, str) and bool(run_id), "Missing nonempty status run_id")
    require(all(row.get("run_id") == run_id for row in phases if row.get("event") != "benchmark_start"),
            "Phase run_id differs from status run_id")
    require(benchmark_start.get("run_id") in (None, run_id), "Benchmark start run_id differs")
    require(all(row.get("run_id") == run_id for row in metrics), "Metric run_id differs from status run_id")
    result = benchmark_end.get("result")
    require(isinstance(result, dict) and result.get("run_id") == run_id
            and result.get("state") == "completed" and result.get("transitions") == expected_transitions,
            "Benchmark final result identity or budget differs")
    recorded_output = benchmark_start["provenance"].get("output_dir")
    configured_output = config.get("output_dir")
    require(isinstance(recorded_output, str) and isinstance(configured_output, str)
            and Path(recorded_output).is_absolute() and Path(configured_output).is_absolute(),
            "Missing absolute recorded/configured training output directory")
    # Remote Wall logs are intentionally copied locally for analysis. Bind the
    # original recorded output to the saved config, not to the local mirror.
    lexical_equal = posixpath.normpath(recorded_output) == posixpath.normpath(configured_output)
    # A local launcher can save a symlink spelling in config while the wrapper
    # records its resolved path. Only accept that exception when both actual
    # directories exist and the filesystem proves that they are the same.
    existing_same_directory = (Path(recorded_output).is_dir() and Path(configured_output).is_dir()
                               and Path(recorded_output).samefile(configured_output))
    require(lexical_equal or existing_same_directory,
            "Recorded output_dir differs from saved config output_dir")
    output_binding = dict(method="lexical_posix" if lexical_equal else "existing_same_directory",
                          configured_output_dir=configured_output, recorded_output_dir=recorded_output)
    if existing_same_directory:
        output_binding["resolved_directory"] = str(Path(recorded_output).resolve())
    require(benchmark_end.get("outcome") == "completed" and not benchmark_end.get("incomplete_batch"),
            "Benchmark failed or has an incomplete final batch")
    require(benchmark_end["transitions"] == expected_transitions, "Benchmark end budget differs")
    require(benchmark_start["provenance"].get("skip_checkpoints") is True,
            "Benchmark must explicitly skip checkpoints")
    require(len({row["session_id"] for row in phases}) == 1, "Mixed benchmark sessions")
    require(benchmark_start["transitions"] == 0 and benchmark_start["updates"] == 0,
            "Benchmark did not start fresh")
    spans = validate_spans(phases)
    rpc = [row for row in spans if row["kind"] == "rpc"]
    updates = [row for row in spans if row["kind"] == "update"]
    waits = [row for row in spans if row["kind"] == "rpc_wait"]
    require(all(row.get("updates_delta") == 1 for row in updates), "Learner call did not perform exactly one Q update")
    require(Counter((row["node"], row["path"]) for row in rpc)
            == Counter((row["node"], row["path"]) for row in waits), "RPC worker and caller counts differ")
    require(not any("publish" in row["path"] or "weight" in row["path"] for row in rpc),
            "DSRL must not publish original model weights")
    batches = sorted((row for row in phases if row.get("event") == "batch_end"),
                     key=lambda row: row["monotonic_seconds"])
    batch_starts = sorted((row for row in phases if row.get("event") == "batch_start"),
                          key=lambda row: row["monotonic_seconds"])
    require(len(batches) == len(batch_starts) == expected_transitions // 64, "Missing or extra complete batches")
    running = sorted((row for row in metrics if row.get("state") == "running"), key=lambda row: row["transitions"])
    require([row["transitions"] for row in running] == list(range(64, expected_transitions + 1, 64)),
            "Running metrics do not cover every completed batch exactly once")
    metric_by_t = {row["transitions"]: row for row in running}
    update_cfg, learner_cfg = config["update"], config["learner"]
    learning_starts = update_cfg["learning_starts"]
    utd = update_cfg["utd"]
    policy_freq = learner_cfg.get("policy_freq", 2)
    auto_alpha = learner_cfg.get("auto_alpha", True)
    require(learning_starts >= 0 and utd > 0 and policy_freq > 0, "Invalid learning schedule")
    generated = previous_t = previous_u = 0
    random_transitions = 0
    for index, (before, after) in enumerate(zip(batch_starts, batches), 1):
        transitions = index * 64
        require(after["batch_index"] == index and after["transitions"] == transitions,
                "Non-contiguous batch boundaries")
        require(after["selected_envs"] == after["transitions_delta"] == before["selected_envs"] == 64,
                "Batch is not 64 real transitions")
        require(before["transitions"] == previous_t and before["updates"] == previous_u,
                "Unexpected transitions or learning outside complete batches")
        require(before["monotonic_seconds"] <= after["monotonic_seconds"]
                and (index == 1 or batches[index-2]["monotonic_seconds"] <= before["monotonic_seconds"]),
                "Overlapping or reversed batch boundaries")
        close(after["start_monotonic_seconds"], before["monotonic_seconds"], "Batch start identity")
        require(after.get("outcome") == "completed", "Failed batch")
        if previous_t < learning_starts:
            random_transitions += 64
        if transitions >= learning_starts:
            generated += 64 * utd
        require(after["updates_delta"] == after["updates"] - previous_u, "Batch update count inconsistent")
        close(after["updates"] + after["update_credit"] + after["dropped_update_credit"], generated,
              f"Update credit identity at transition {transitions}")
        require(0 <= after["update_credit"] <= update_cfg["max_backlog"] and after["dropped_update_credit"] >= 0,
                "Invalid backlog or dropped credit")
        metric = metric_by_t[transitions]
        for phase_key, metric_key in (("updates", "updates"), ("episodes", "episodes"),
                                     ("update_credit", "update_backlog"),
                                     ("dropped_update_credit", "dropped_update_credit")):
            close(after[phase_key], metric[metric_key], f"Boundary and metric {phase_key}")
        in_batch = [row for row in updates if span_bounds(row)[0] >= before["monotonic_seconds"]
                    and span_bounds(row)[1] <= after["monotonic_seconds"]]
        require(len(in_batch) == after["updates_delta"], "Update spans do not match batch optimizer delta")
        previous_t, previous_u = transitions, after["updates"]
    initial = unique_event(phases, "learner_initial")["identity"]
    final = unique_event(phases, "learner_final")["identity"]
    require(initial is not None and final is not None, "Missing learner identity receipt")
    require(initial["updates"] == 0 and final["updates"] == previous_u == len(updates) == status["updates"],
            "Final updates differ from real learner calls or initial state")
    require(benchmark_end["updates"] == previous_u, "Unaccounted post-loop updates")
    require(previous_u > 0 and initial["actor_sha256"] != final["actor_sha256"], "Actor did not change")
    for name in ("q_optimizer", "actor_optimizer", "alpha_optimizer"):
        optimizer_steps(initial, name, 0)
    expected_actor = previous_u // policy_freq
    expected_alpha = expected_actor if auto_alpha else 0
    observed_optimizers = {name: optimizer_steps(final, name, count) for name, count in
                          (("q_optimizer", previous_u), ("actor_optimizer", expected_actor),
                           ("alpha_optimizer", expected_alpha))}
    require(expected_actor > 0, "No real actor optimizer step")
    for field, metric_key in (("update_credit", "update_backlog"), ("dropped_update_credit", "dropped_update_credit")):
        close(benchmark_end[field], status[metric_key], f"Final {field}")
        close(batches[-1][field], status[metric_key], f"Final boundary {field}")
    save_spans = [row for row in spans if row["method"] == "_save"]
    skipped = [row for row in phases if row.get("event") == "checkpoint_skipped"]
    require(len(save_spans) == len(skipped) == len(batches) + 1
            and all(row.get("skipped") is True for row in save_spans)
            and sum(bool(row.get("final")) for row in skipped) == 1,
            "All per-batch and final saves must be explicitly skipped")
    checkpoint_files = [path for path in training_dir.rglob("*") if path.is_file()
                        and path.suffix in (".pt", ".pth", ".ckpt", ".safetensors")]
    require(not checkpoint_files and not status.get("last_checkpoint"), "Unexpected checkpoint payload")
    require(len(episodes) == status["episodes"] == batches[-1]["episodes"], "Episode count differs from log")
    start_row = next(row for row in batches if row["transitions"] == start_transition)
    end_row = batches[-1]
    start, end = start_row["monotonic_seconds"], end_row["monotonic_seconds"]
    require(end > start, "Matched window has no positive elapsed time")
    last_metric = running[-1]
    full_elapsed = last_metric["elapsed_seconds"]
    require(math.isfinite(full_elapsed) and full_elapsed > 0, "Invalid full-loop native elapsed time")
    full_start = end - full_elapsed
    partition, routes, caller_waits = timing_window(start, end, rpc, updates, waits)
    full_partition, full_routes, full_waits = timing_window(full_start, end, rpc, updates, waits)
    window_updates = [row for row in updates if clipped(row, start, end) > 0]
    require(all(span_bounds(row)[0] >= start and span_bounds(row)[1] <= end for row in window_updates),
            "Learner update crosses a completed-batch boundary")
    start_u, end_u = start_row["updates"], end_row["updates"]
    require(len(window_updates) == end_u - start_u, "Matched window update count differs")

    def work(first_u, last_u, transitions, credit, backlog, dropped):
        actor = last_u // policy_freq - first_u // policy_freq
        return dict(q_optimizer_steps=last_u-first_u, actor_optimizer_steps=actor,
                    alpha_optimizer_steps=actor if auto_alpha else 0,
                    q_ensemble_size=learner_cfg.get("num_qs", 5),
                    optimizer_minibatch_size=learner_cfg["batch_size"],
                    update_counter_start=first_u, update_counter_end=last_u,
                    actual_q_updates_per_transition=(last_u-first_u)/transitions,
                    generated_update_credit=credit, backlog_delta=backlog, dropped_credit_delta=dropped)

    full_work = work(0, end_u, expected_transitions, generated,
                     end_row["update_credit"], end_row["dropped_update_credit"])
    matched_credit = sum(row["transitions_delta"] * utd for row in batches
                         if row["transitions"] > start_transition and row["transitions"] >= learning_starts)
    matched_work = work(start_u, end_u, expected_transitions-start_transition, matched_credit,
        end_row["update_credit"]-start_row["update_credit"],
        end_row["dropped_update_credit"]-start_row["dropped_update_credit"])
    close(matched_work["q_optimizer_steps"]+matched_work["backlog_delta"]+matched_work["dropped_credit_delta"],
          matched_credit, "Matched-window credit conservation")

    def workload(first_t, last_t, selected_routes):
        before = metric_by_t[first_t] if first_t else {}
        after = metric_by_t[last_t]
        selected = [row for row in episodes if first_t < row["transition"] <= last_t]
        counts = dict(episodes=len(selected), successes=sum(bool(row.get("success")) for row in selected),
                      terminated=sum(bool(row.get("terminated")) for row in selected),
                      truncated=sum(bool(row.get("truncated")) for row in selected),
                      reset_calls=after["reset_calls"]-before.get("reset_calls", 0),
                      reset_envs=after["reset_envs"]-before.get("reset_envs", 0),
                      simulation_step_calls=sum(value["count"] for route,value in selected_routes.items()
                                                if route in ("simulation:/step", "simulation:/step_and_reset")))
        close(counts["episodes"], after["episodes"]-before.get("episodes",0), "Window episode count")
        close(counts["successes"], after["successes"]-before.get("successes",0), "Window success count")
        close(counts["simulation_step_calls"], (last_t-first_t)//64, "Window simulation step count")
        if not config["rollout"].get("fused_step_reset",False):
            close(counts["reset_calls"], selected_routes.get("simulation:/reset",{}).get("count",0),
                  "Window reset RPC count")
        return counts

    provenance = benchmark_start["provenance"]
    raw_paths = [phases_path, *[path for path in paths.values() if path.exists()], Path(__file__).resolve(), WRAPPER]
    raw_paths += [training_dir/name for name in ("rpc_timing.jsonl", "initialization.json", "events.jsonl")
                  if (training_dir/name).exists()]
    full = dict(transitions=expected_transitions, elapsed_seconds=full_elapsed,
                transitions_per_second=expected_transitions/full_elapsed,
                timing_source="Last completed running metrics.elapsed_seconds; services already ready; shutdown excluded",
                start_monotonic_seconds=full_start, end_monotonic_seconds=end,
                start_unix=end_row["unix_seconds"]-full_elapsed, end_unix=end_row["unix_seconds"],
                interval_start_is_derived=True,
                interval_alignment_note="Full interval start is derived by anchoring native elapsed to the final post-status batch marker; native timer ends slightly before that marker. Matched window uses exact markers.",
                async_partition=full_partition, rpc_routes=full_routes, caller_waits=full_waits)
    matched = dict(start_transition=start_transition, end_transition=expected_transitions,
                   transitions=expected_transitions-start_transition,
                   start_monotonic_seconds=start, end_monotonic_seconds=end,
                   start_unix=start_row["unix_seconds"], end_unix=end_row["unix_seconds"],
                   elapsed_seconds=end-start, transitions_per_second=(expected_transitions-start_transition)/(end-start),
                   timing_source="Exact completed batch_end to batch_end monotonic interval")
    recording = dict(full_reported_seconds=end_row.get("recording_seconds",0.),
                     matched_reported_seconds=end_row.get("recording_seconds",0.)-start_row.get("recording_seconds",0.),
                     final_reported_seconds=benchmark_end.get("recording_seconds",0.),
                     subtracted_from_wall_time=False,
                     semantics="Wrapper cumulative record() cost; excludes the currently emitted record, queue costs and some instrumentation bookkeeping; not total observer overhead")
    return dict(schema="pressb.dsrl_benchmark.summary.v1", full_loop=full, matched_window=matched,
                async_partition=partition, rpc_routes=routes, caller_waits=caller_waits,
                learning_work=dict(full=full_work, matched_window=matched_work,
                    configured_utd=utd, policy_freq=policy_freq, auto_alpha=auto_alpha,
                    random_warmup_transitions=random_transitions,
                    final_update_backlog=end_row["update_credit"],
                    final_dropped_update_credit=end_row["dropped_update_credit"],
                    final_optimizer_receipts=observed_optimizers,
                    q_parameter_hash_recorded=False,
                    q_parameter_change_evidence="Real Q optimizer parameter-state steps; the frozen wrapper did not record a Q parameter hash",
                    actor_sha256_initial=initial["actor_sha256"], actor_sha256_final=final["actor_sha256"]),
                workload=dict(full=workload(0,expected_transitions,full_routes),
                              matched_window=workload(start_transition,expected_transitions,routes)),
                recording_overhead=recording,
                provenance=dict(benchmark=provenance, training_dir=str(training_dir), phases=str(phases_path),
                                recorded_training_output_dir=recorded_output,
                                output_dir_binding=output_binding,
                                local_artifact_copy=Path(recorded_output).resolve() != training_dir,
                                relocated_artifacts=dict(original_output_dir=recorded_output,
                                    current_training_dir=str(training_dir),
                                    relocated=Path(recorded_output).resolve() != training_dir),
                                run_id=status.get("run_id"), session_id=benchmark_start["session_id"],
                                reported_source_identities_not_reloaded=True),
                source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in raw_paths},
                audit=dict(verdict="passed", exact_budget=True, complete_64_transition_batches=len(batches),
                           no_failed_or_incomplete_spans=True, actor_changed=True,
                           actual_optimizer_steps_verified=True, update_credit_conserved=True,
                           checkpoint_saves_skipped=len(save_spans), checkpoint_payloads=0,
                           span_order="Explicit monotonic interval bounds, independent of JSONL/event_id order",
                           wall_partition="RPC-only + learner-only + overlap + neither; inclusive route and caller costs not added",
                           frozen_base_weights="No weight-publish RPC observed; original model checkpoint identity is an external experiment receipt, not inferred from this trainer log"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--phases", type=Path, required=True)
    parser.add_argument("--start-transition", type=int, required=True)
    parser.add_argument("--expected-transitions", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize(args.training_dir, args.phases, args.start_transition, args.expected_transitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.output), audit=report["audit"], matched_window=report["matched_window"])))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, KeyError, OSError) as error:
        print(f"DSRL benchmark audit failed: {error}", file=sys.stderr)
        raise SystemExit(2)
