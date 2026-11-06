#!/usr/bin/env python3
"""Measure a real three-node training window without contacting any service.

One transition is one environment action chunk, never a camera frame or tick.
RPC durations include overlapped learner work and must not be added to learner
seconds. The default window starts only after warmup and actual SAC updates.
This tool imports neither Isaac nor Torch and does not allocate GPU resources.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


MONOTONIC = ("transitions", "updates", "episodes", "successes", "elapsed_seconds",
             "dropped_update_credit", "learner_seconds")
TERMINAL_STATES = {"completed", "stopped", "failed", "transition_limit"}
RESET_COUNTERS = ("reset_envs", "reset_calls")


def read_jsonl(path):
    """Ignore only an incomplete final live-write record, never an interior error."""
    raw = Path(path).read_bytes()
    lines = raw.splitlines(keepends=True)
    rows, ignored = [], False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index == len(lines)-1 and not line.endswith(b"\n"):
                ignored = True
                break
            raise ValueError(f"Invalid JSON in {path}, line {index+1}") from None
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object in {path}, line {index+1}")
        rows.append(row)
    return rows, {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                  "bytes": len(raw), "ignored_incomplete_final_line": ignored}


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), {"path": str(Path(path).resolve()),
                             "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def number(row, key, default=None):
    value = row.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing or non-finite numeric counter: {key}")
    return value


def validate_metrics(rows):
    if len(rows) < 2:
        raise ValueError("At least two complete metric records are required")
    identity = (rows[0].get("run_id"), rows[0].get("schema_fingerprint"))
    if not identity[0]:
        raise ValueError("Metrics must identify a run_id")
    reset_counters_present = any(key in rows[0] for key in RESET_COUNTERS)
    previous = None
    for row in rows:
        if (row.get("run_id"), row.get("schema_fingerprint")) != identity:
            raise ValueError("Mixed run identities; analyze each session separately")
        for key in (*MONOTONIC, "update_backlog"):
            value = number(row, key)
            if value < 0 or (previous and key in MONOTONIC and value < previous[key]):
                raise ValueError(f"Counter regressed or became negative: {key}; do not merge sessions")
        for field in ("rpc_seconds", "rpc_calls"):
            for key in set(row.get(field, {})) | set(previous.get(field, {}) if previous else {}):
                value = number(row.get(field, {}), key, 0)
                prior = number(previous.get(field, {}), key, 0) if previous else 0
                if value < prior:
                    raise ValueError(f"Counter regressed: {field}.{key}")
        # New trainers count only reset rows accepted after validation. A fused
        # step may reset zero environments, so its RPC count is not evidence.
        if any(key in row for key in RESET_COUNTERS) != reset_counters_present:
            raise ValueError("Reset counter schema changed within a session")
        if reset_counters_present:
            for key in RESET_COUNTERS:
                value = row.get(key)
                if type(value) is not int or value < 0 or (previous and value < previous[key]):
                    raise ValueError(f"Missing, invalid or regressed reset counter: {key}")
            if row["reset_calls"] > row["reset_envs"]:
                raise ValueError("Actual reset calls cannot exceed reset environments")
        previous = row


def delta(end, start, key):
    return number(end, key)-number(start, key)


def reset_evidence(start, end, rpc):
    explicit_calls = rpc.get("simulation/reset", {}).get("calls", 0)
    fused_calls = rpc.get("simulation/step_and_reset", {}).get("calls", 0)
    if "reset_envs" in start:
        environments = delta(end, start, "reset_envs")
        actual_calls = delta(end, start, "reset_calls")
        if actual_calls > environments or bool(actual_calls) != bool(environments):
            raise ValueError("Reset counter deltas do not identify nonempty reset batches")
        observed = environments > 0 and actual_calls > 0
        source = "validated_reset_items_counters"
    else:
        # Legacy serial /reset callers emit metrics only after validating a
        # successful response. Never infer resets from /step_and_reset alone.
        environments, actual_calls = None, explicit_calls
        observed = explicit_calls > 0
        source = "legacy_explicit_reset_rpc" if explicit_calls else "no_verified_reset_evidence"
    return {"observed": observed, "evidence_source": source,
            "environment_resets": environments, "actual_reset_batches": actual_calls,
            "explicit_reset_rpc_calls": explicit_calls, "fused_step_rpc_calls": fused_calls,
            "fused_rpc_call_count_is_not_a_reset_count": True}


def analyze(rows, episodes, config, *, last_seconds=None, after_transition=None,
            target_rate=15., min_seconds=300., min_transitions=5000):
    validate_metrics(rows)
    if config.get("mode", "train") != "train":
        raise ValueError("Training throughput qualification requires mode=train")
    warmup = max(config["update"]["learning_starts"], config["learner"].get("batch_size", 256))
    threshold = max(warmup, after_transition or 0)
    candidates = [row for row in rows if row["transitions"] >= threshold and row["updates"] > 0]
    if last_seconds is not None:
        if not math.isfinite(last_seconds) or last_seconds <= 0:
            raise ValueError("last_seconds must be positive and finite")
        cutoff = rows[-1]["elapsed_seconds"]-last_seconds
        candidates = [row for row in candidates if row["elapsed_seconds"] >= cutoff]
    if len(candidates) < 2:
        raise ValueError("Need two records after warmup/selected window with real learner updates")
    start, end = candidates[0], candidates[-1]
    seconds = delta(end, start, "elapsed_seconds")
    transitions = delta(end, start, "transitions")
    if seconds <= 0 or transitions <= 0:
        raise ValueError("Selected window must advance both wall time and transitions")
    rate = transitions/seconds
    updates = delta(end, start, "updates")
    utd = number(config["update"], "utd")
    if utd < 0:
        raise ValueError("UTD cannot be negative")
    backlog = delta(end, start, "update_backlog")
    dropped = delta(end, start, "dropped_update_credit")
    expected = transitions*utd
    credit_error = updates+backlog+dropped-expected
    selected_episodes = [row for row in episodes
                         if start["transitions"] < row["transition"] <= end["transitions"]]
    episode_ids = [(row.get("env_id"), row.get("episode_id")) for row in selected_episodes]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Duplicate completed episode identities")
    reasons = Counter(row.get("info", {}).get("termination_reason", "unspecified")
                      for row in selected_episodes)
    successes = sum(row.get("success") is True for row in selected_episodes)
    timeouts = sum(row.get("truncated") is True or row.get("info", {}).get("termination_reason") == "timeout"
                   for row in selected_episodes)
    episodes_match = (len(selected_episodes) == delta(end, start, "episodes")
                      and successes == delta(end, start, "successes"))
    rpc = {}
    rpc_names = {key for row in (start, end) for field in ("rpc_seconds", "rpc_calls")
                 for key in row.get(field, {})}
    for name in sorted(rpc_names):
        duration = end.get("rpc_seconds", {}).get(name, 0)-start.get("rpc_seconds", {}).get(name, 0)
        calls = end.get("rpc_calls", {}).get(name, 0)-start.get("rpc_calls", {}).get(name, 0)
        rpc[name] = {"seconds": duration, "calls": calls,
                     "seconds_per_call": duration/calls if calls else None,
                     "duration_over_wall": duration/seconds}
    resets = reset_evidence(start, end, rpc)
    finite_losses = all(bool(row.get("last_update_metrics")) and all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        for value in row["last_update_metrics"].values()) for row in candidates)
    gates = {"sufficient_wall_time": seconds >= min_seconds,
             "sufficient_transitions": transitions >= min_transitions,
             "target_throughput": rate >= target_rate,
             "configured_utd_one": math.isclose(utd, 1.),
             "completed_utd_at_least_098": updates/transitions >= .98,
             "no_dropped_update_credit": dropped == 0,
             "update_accounting_balances": abs(credit_error) < 1e-6,
             "finite_learner_metrics": finite_losses,
             "episode_log_matches_counters": episodes_match,
             "success_and_timeout_observed": successes > 0 and timeouts > 0,
             "indexed_resets_observed": resets["observed"],
             "no_failed_run": end.get("state") != "failed"}
    return {"schema_version": 1, "scope": "measured_end_to_end_post_warmup_training_window",
        "run_id": end["run_id"], "schema_fingerprint": end.get("schema_fingerprint"),
        "unit": "one environment action chunk is one transition",
        "window": {"start_transitions": start["transitions"], "end_transitions": end["transitions"],
                   "start_elapsed_seconds": start["elapsed_seconds"], "end_elapsed_seconds": end["elapsed_seconds"],
                   "wall_seconds": seconds, "transitions": transitions, "transitions_per_second": rate,
                   "metric_records": len(candidates), "end_state": end.get("state"),
                   "includes_final_status_time": end.get("_final_status_appended", False)},
        "learning": {"updates": updates, "configured_utd": utd, "completed_utd": updates/transitions,
                     "backlog_start": start["update_backlog"], "backlog_end": end["update_backlog"],
                     "new_dropped_update_credit": dropped, "credit_accounting_error": credit_error,
                     "learner_seconds": delta(end, start, "learner_seconds"),
                     "learner_seconds_overlap_rpc": True},
        "episodes": {"completed": len(selected_episodes), "successes": successes, "timeouts": timeouts,
                     "termination_reasons": dict(reasons), "success_rate": successes/len(selected_episodes)
                     if selected_episodes else None, "counter_match": episodes_match},
        "rpc": rpc, "resets": resets,
        "projection": {str(count): {"hours": count/rate/3600,
                                     "remaining_hours": max(0, count-end["transitions"])/rate/3600}
                       for count in (400000, 1000000)},
        "qualification": {"passed": all(gates.values()), "checks": gates,
                          "target_transitions_per_second": target_rate,
                          "minimum_seconds": min_seconds, "minimum_transitions": min_transitions},
        "limitations": ["Projected hours are extrapolations, not completed-run durations.",
            "RPC call durations can overlap learner work or each other; never sum them with learner time.",
            "Window includes resets, transfers, inference and periodic checkpoints that occurred inside it.",
            "Model/simulator startup and pre-window warmup are excluded from this steady training window.",
            "Correct camera images, reset isolation, randomized coverage and contact physics need separate qualification."]}


def resource_summary(rows):
    """Use capacity-probe resource schema; no implicit alignment of separate clocks."""
    devices = {}
    for row in rows:
        for device in row.get("gpu", {}).get("devices", []):
            key = device.get("uuid", str(device["index"]))
            entry = devices.setdefault(key, {"index": device["index"], "peak_used_mib": 0.,
                                             "minimum_free_mib": math.inf})
            entry["peak_used_mib"] = max(entry["peak_used_mib"], number(device, "used_mib"))
            entry["minimum_free_mib"] = min(entry["minimum_free_mib"], number(device, "free_mib"))
    def extrema(key, function):
        values = [number(row, key) for row in rows if key in row]
        return function(values) if values else None
    return {"scope": "entire_resource_file_not_time_aligned_to_training_window", "samples": len(rows),
            "devices": devices, "minimum_host_available_bytes": extrema("host_available_bytes", min),
            "peak_process_tree_rss_bytes": extrema("process_tree_rss_bytes", max),
            "measurement_errors": [error for row in rows for error in row.get("gpu", {}).get("errors", [])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Trainer output containing config/metrics/episodes")
    parser.add_argument("--last-seconds", type=float, help="Use existing sample boundaries inside this trailing window")
    parser.add_argument("--after-transition", type=int, help="Additional lower bound, always after warmup")
    parser.add_argument("--target-rate", type=float, default=15.)
    parser.add_argument("--min-seconds", type=float, default=300.)
    parser.add_argument("--min-transitions", type=int, default=5000)
    parser.add_argument("--resources", type=Path, help="Optional benchmark_pap_parallel resource JSONL")
    parser.add_argument("--output", type=Path, help="Otherwise print JSON; never modifies training files")
    args = parser.parse_args()
    for key in ("target_rate", "min_seconds", "min_transitions"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive and finite")
    rows, metrics_source = read_jsonl(args.run / "metrics.jsonl")
    episodes, episodes_source = read_jsonl(args.run / "episodes.jsonl")
    config, config_source = read_json(args.run / "config.json")
    sources = {"metrics": metrics_source, "episodes": episodes_source, "config": config_source}
    status_path = args.run / "status.json"
    if rows and status_path.exists():
        status, status_source = read_json(status_path)
        # A final checkpoint may finish after the last metrics row. Include its
        # real wall duration instead of making end-of-run performance look faster.
        if (status.get("state") in TERMINAL_STATES and status.get("run_id") == rows[-1].get("run_id")
                and status.get("elapsed_seconds", -1) > rows[-1].get("elapsed_seconds", 0)):
            rows.append({**status, "_final_status_appended": True})
            sources["final_status"] = status_source
    report = analyze(rows, episodes, config, last_seconds=args.last_seconds,
                     after_transition=args.after_transition, target_rate=args.target_rate,
                     min_seconds=args.min_seconds, min_transitions=args.min_transitions)
    report["sources"] = sources
    if args.resources:
        samples, source = read_jsonl(args.resources)
        report["resources"] = resource_summary(samples)
        report["sources"]["resources"] = source
    payload = json.dumps(report, indent=2, allow_nan=False)+"\n"
    if args.output:
        if args.output.resolve() in {path.resolve() for path in args.run.glob("*") if path.is_file()}:
            parser.error("Refusing to overwrite a trainer output file")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
