#!/usr/bin/env python3
"""Read-only host/GPU/PID sampling; runs standalone on Linux, including H200.

Requires psutil and optionally nvidia-smi, never imports CUDA/Isaac or calls RPC.
All memory fields use bytes unless named *_mib; counters are cumulative. Host
and GPU values include unrelated users. Process CPU 100% means one CPU core.

Collect (copy this single file to a remote host if needed)::
  python3 scripts/monitor_runtime_resources.py collect --output resources.jsonl \
    --interval 2 --duration 1800 --status run/status.json \
    --pid trainer=12345 --pid-file inference=inference_process.json \
    --phase-file benchmark_phase.json

The optional marker is a JSON object, e.g. {"phase":"steady"} or
{"phase":"checkpoint"}. The benchmark driver must update it atomically at
real phase boundaries. Without a marker, status.state is observed but is NOT
labelled steady: DPPO's "updating" also includes publication and checkpoint IO.

Summarize explicit steady windows (gaps/phase changes are never bridged)::
  python3 scripts/monitor_runtime_resources.py summarize resources.jsonl \
    --phase steady --min-transitions 1024 --output resources_summary.json

Samples have UTC and host monotonic clocks, and status file mtime/age plus
trainer elapsed/counters. Cross-host UTC needs synchronized clocks; monotonic
clocks are local. Sampled maxima may miss transient peaks. Marker boundaries
have sampling uncertainty, not exact profiler timing. Ctrl-C/SIGTERM stops only
the monitor; no monitored process is signalled. Existing output is not replaced.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time

import psutil


STATUS_KEYS = ("state", "phase", "run_id", "method", "mode", "transitions", "updates",
               "iterations", "actor_updates", "critic_updates", "policy_version",
               "episodes", "successes", "elapsed_seconds", "last_checkpoint",
               "critic_warmup_iterations_remaining")


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def gpu_snapshot(timeout=2.):
    """N/A values remain null; device and compute queries fail independently."""
    queries = {
        "devices": ("--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,"
                    "utilization.memory,power.draw,temperature.gpu", [
                        "index", "uuid", "name", "memory_used_mib", "memory_total_mib",
                        "gpu_utilization_percent", "memory_utilization_percent", "power_w", "temperature_c"]),
        "processes": ("--query-compute-apps=gpu_uuid,pid,used_memory", ["gpu_uuid", "pid", "memory_used_mib"]),
    }
    result = {"devices": [], "processes": [], "errors": []}
    for kind, (query, keys) in queries.items():
        try:
            run = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, check=True, timeout=timeout)
            for parts in csv.reader(io.StringIO(run.stdout), skipinitialspace=True):
                if not parts:
                    continue
                if len(parts) != len(keys):
                    result["errors"].append(f"{kind}: unexpected CSV row {parts!r}")
                    continue
                row = {key: value.strip() if key in ("uuid", "gpu_uuid", "name") else number(value)
                       for key, value in zip(keys, parts)}
                for key in ("pid", "index"):
                    if key in row and row[key] is not None:
                        row[key] = int(row[key])
                result[kind].append(row)
        except (OSError, subprocess.SubprocessError) as error:
            result["errors"].append(f"{kind}: {type(error).__name__}: {error}")
    return result


def read_json_file(path, now):
    if path is None:
        return None
    result = {"path": str(path), "data": None}
    try:
        # fstat refers to the opened version even if its pathname is replaced.
        with Path(path).open() as stream:
            result["mtime_unix"] = os.fstat(stream.fileno()).st_mtime
            result["age_seconds"] = now - result["mtime_unix"]
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        result["data"] = data
    except (OSError, ValueError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def phase_for(status, marker):
    if marker and isinstance((marker.get("data") or {}).get("phase"), str) and marker["data"]["phase"]:
        return {"name": str(marker["data"]["phase"]), "source": "marker", "explicit": True}
    if marker is not None:
        # Never silently continue a steady window through an unreadable marker.
        return {"name": "marker_unavailable", "source": "marker", "explicit": False}
    data = (status or {}).get("data") or {}
    if data.get("phase"):
        return {"name": str(data["phase"]), "source": "status.phase", "explicit": True}
    return {"name": str(data.get("state", "unobserved")), "source": "status.state", "explicit": False}


def start_ticks(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


class ProcessSampler:
    """Bind root identities; do not attach reused root PIDs. No signals sent."""

    def __init__(self, pids=(), pid_files=()):
        self.pids = dict(pids)
        self.pid_files = dict(pid_files)
        self.bound = {}
        self.tracked = {}
        self.previous = {}

    def sample(self, gpu_processes, now_mono):
        errors, owners = [], {}
        for role in self.pids.keys() | self.pid_files.keys():
            receipt = {"pid": self.pids.get(role)}
            try:
                if role in self.pid_files:
                    candidate = json.loads(Path(self.pid_files[role]).read_text())
                    if not isinstance(candidate, dict):
                        raise ValueError("receipt must be a JSON object")
                    receipt = candidate
                pid = receipt["pid"]
                if type(pid) is not int or pid <= 0:
                    raise ValueError("receipt must contain a positive integer pid")
                parent = psutil.Process(pid)
                identity = (pid, parent.create_time())
                if "start_ticks" in receipt and str(receipt["start_ticks"]) != start_ticks(pid):
                    raise ValueError("receipt start_ticks does not match live PID")
                expected = receipt.get("create_time")
                if expected is not None and float(expected) != identity[1]:
                    raise ValueError("receipt create_time does not match live PID")
                if role not in self.bound:
                    self.bound[role] = identity
                if self.bound[role] != identity:
                    raise ValueError("root PID identity changed; start a new monitor for a new run")
                candidates = [parent, *parent.children(recursive=True)]
                if parent.create_time() != identity[1] or not parent.is_running():
                    raise ValueError("root exited during child discovery")
                owners[role] = {"pid": pid, "create_time": identity[1], "alive": True}
                for process in candidates:
                    key = (process.pid, process.create_time())
                    self.tracked.setdefault(key, set()).add(role)
            except (OSError, ValueError, TypeError, KeyError, psutil.Error) as error:
                owners[role] = {"pid": receipt.get("pid"), "alive": False,
                                "error": f"{type(error).__name__}: {error}"}
        for gpu_process in gpu_processes:
            if gpu_process.get("pid") is None:
                continue
            try:
                process = psutil.Process(gpu_process["pid"])
                self.tracked.setdefault((process.pid, process.create_time()), set()).add("gpu_compute")
            except psutil.Error:
                pass
        rows = []
        for identity, roles in list(self.tracked.items()):
            pid, created = identity
            try:
                process = psutil.Process(pid)
                if process.create_time() != created or not process.is_running():
                    del self.tracked[identity]
                    self.previous.pop(identity, None)
                    continue
                with process.oneshot():
                    cpu = process.cpu_times()
                    seconds = cpu.user + cpu.system
                    row = {"pid": pid, "create_time": created, "roles": sorted(roles),
                           "name": process.name(), "status": process.status(),
                           "rss_bytes": process.memory_info().rss, "cpu_seconds": seconds,
                           "cpu_percent": None}
                    try:
                        row["io_counters"] = process.io_counters()._asdict()
                    except (psutil.Error, AttributeError):
                        row["io_counters"] = None
                prior = self.previous.get(identity)
                if prior and now_mono > prior[0] and seconds >= prior[1]:
                    row["cpu_percent"] = 100 * (seconds - prior[1]) / (now_mono - prior[0])
                self.previous[identity] = (now_mono, seconds)
                rows.append(row)
            except psutil.NoSuchProcess:
                self.tracked.pop(identity, None)
                self.previous.pop(identity, None)
            except psutil.Error as error:
                errors.append(f"pid {pid}: {type(error).__name__}: {error}")
        return {"owners": owners, "rows": rows, "errors": errors,
                "rss_bytes": sum(row["rss_bytes"] for row in rows),
                "role_rss_bytes": {role: sum(row["rss_bytes"] for row in rows if role in row["roles"])
                                   for role in self.pids.keys() | self.pid_files.keys()}}


def host_snapshot(first=False):
    memory, swap = psutil.virtual_memory(), psutil.swap_memory()
    cpu = psutil.cpu_percent(interval=None)
    disk, network = psutil.disk_io_counters(), psutil.net_io_counters()
    return {"cpu_percent": None if first else cpu, "cpu_count": psutil.cpu_count(),
            "load_average": list(os.getloadavg()), "memory_total_bytes": memory.total,
            "memory_available_bytes": memory.available, "memory_used_bytes": memory.used,
            "swap_total_bytes": swap.total, "swap_used_bytes": swap.used,
            "swap_in_bytes": swap.sin, "swap_out_bytes": swap.sout,
            "network_counters": network._asdict() if network else None,
            "disk_counters": disk._asdict() if disk else None}


def collect(args):
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    processes = ProcessSampler(args.pid, args.pid_file)
    started = time.monotonic()
    hostname, sample_index = socket.gethostname(), 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", buffering=1) as stream:
        while not stopped and time.monotonic() - started < args.duration:
            tick = time.monotonic()
            now = time.time()
            status, marker = read_json_file(args.status, now), read_json_file(args.phase_file, now)
            if status and status.get("data"):
                status["data"] = {key: status["data"][key] for key in STATUS_KEYS if key in status["data"]}
            gpu = gpu_snapshot(args.query_timeout)
            record = {"schema_version": 1, "hostname": hostname,
                      "utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                      "wall_time": now, "monotonic_seconds": tick,
                      "monitor_elapsed_seconds": tick - started, "sample_index": sample_index,
                      "requested_interval_seconds": args.interval, "status": status, "marker": marker,
                      "phase": phase_for(status, marker), "gpu": gpu,
                      "host": host_snapshot(first=sample_index == 0),
                      "processes": processes.sample(gpu["processes"], time.monotonic())}
            record["sample_duration_seconds"] = time.monotonic() - tick
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            sample_index += 1
            deadline = min(started + args.duration, tick + args.interval)
            while not stopped and time.monotonic() < deadline:
                time.sleep(min(.2, max(0., deadline - time.monotonic())))
    print(json.dumps({"state": "stopped" if stopped else "completed", "samples": sample_index,
                      "output": str(args.output), "elapsed_seconds": time.monotonic() - started}))


def stats(values):
    values = sorted(value for value in values if number(value) is not None)
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None, "p95": None}
    return {"count": len(values), "min": values[0], "mean": statistics.mean(values),
            "max": values[-1], "p95": values[max(0, math.ceil(.95 * len(values)) - 1)]}


def summarize(records, phases=(), min_transitions=0, max_gap_seconds=5., start_unix=None, end_unix=None,
              gpu_selectors=(), roles=()):
    """Aggregate samples and contiguous counter windows; never bridge exclusions."""
    selected, windows, current = [], [], []
    excluded = {}

    def flush():
        nonlocal current
        if current:
            windows.append(current)
            current = []

    for row in records:
        status = (row.get("status") or {}).get("data") or {}
        phase = row["phase"]
        reason = None
        if ((start_unix is not None and row["wall_time"] < start_unix)
                or (end_unix is not None and row["wall_time"] > end_unix)):
            reason = "time_window"
        elif phases and phase["name"] not in phases:
            reason = "phase_filter"
        elif phases and phase["name"] in ("steady", "steady_state") and not phase["explicit"]:
            reason = "implicit_steady_phase"
        elif min_transitions and (number(status.get("transitions")) is None
                                  or status["transitions"] < min_transitions):
            reason = "transition_warmup"
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
            flush()
            continue
        if current:
            before = current[-1]
            old = (before.get("status") or {}).get("data") or {}
            gap = row["monotonic_seconds"] - before["monotonic_seconds"]
            counters_changed = False
            for key in ("transitions", "elapsed_seconds"):
                old_value, new_value = number(old.get(key)), number(status.get(key))
                counters_changed |= ((old_value is None) != (new_value is None)
                                     or (old_value is not None and new_value is not None and new_value < old_value))
            # A short excluded phase may occur entirely between samples. A new
            # marker file version proves the phase instance changed even if its
            # name is identical; never bridge that unobserved interval.
            marker_changed = (phase["source"] == "marker"
                              and (row.get("marker") or {}).get("mtime_unix")
                              != (before.get("marker") or {}).get("mtime_unix"))
            if (row["hostname"] != before["hostname"] or phase != before["phase"]
                    or status.get("run_id") != old.get("run_id") or not 0 < gap <= max_gap_seconds
                    or counters_changed or marker_changed):
                flush()
        current.append(row)
        selected.append(row)
    flush()
    interval_reports = []
    for window in windows:
        first, last = window[0], window[-1]
        before = (first.get("status") or {}).get("data") or {}
        after = (last.get("status") or {}).get("data") or {}
        dt = last["monotonic_seconds"] - first["monotonic_seconds"]
        report = {"phase": first["phase"], "run_id": before.get("run_id"), "samples": len(window),
                  "first_wall_time": first["wall_time"], "last_wall_time": last["wall_time"],
                  "observed_seconds": dt, "first_status": before, "last_status": after,
                  "transition_delta": None, "trainer_elapsed_delta": None,
                  "observed_transitions_per_second": None, "trainer_transitions_per_second": None,
                  "max_status_age_seconds": max((r["status"].get("age_seconds", 0)
                                                  for r in window if r.get("status")), default=None)}
        report["host_counter_deltas"] = {}
        for group, keys in ((None, ("swap_in_bytes", "swap_out_bytes")),
                            ("network_counters", ("bytes_sent", "bytes_recv")),
                            ("disk_counters", ("read_bytes", "write_bytes"))):
            start = first["host"] if group is None else (first["host"].get(group) or {})
            end = last["host"] if group is None else (last["host"].get(group) or {})
            for key in keys:
                difference = None
                if number(start.get(key)) is not None and number(end.get(key)) is not None:
                    value = end[key] - start[key]
                    difference = value if value >= 0 else None
                report["host_counter_deltas"][f"{group}.{key}" if group else key] = difference
        if (before.get("run_id") and before.get("run_id") == after.get("run_id")
                and all(number(s.get("transitions")) is not None for s in (before, after))):
            delta = after["transitions"] - before["transitions"]
            report["transition_delta"] = delta
            report["observed_transitions_per_second"] = delta / dt if dt > 0 else None
            if all(number(s.get("elapsed_seconds")) is not None for s in (before, after)):
                elapsed = after["elapsed_seconds"] - before["elapsed_seconds"]
                report["trainer_elapsed_delta"] = elapsed
                report["trainer_transitions_per_second"] = delta / elapsed if elapsed > 0 else None
        interval_reports.append(report)
    gpu_samples = {}
    for row in selected:
        for gpu in row["gpu"]["devices"]:
            if gpu_selectors and not ({gpu["uuid"], str(gpu.get("index"))} & set(gpu_selectors)):
                continue
            value = dict(gpu)
            used, total = number(gpu.get("memory_used_mib")), number(gpu.get("memory_total_mib"))
            value["memory_capacity_used_percent"] = 100 * used / total if used is not None and total else None
            gpu_samples.setdefault(gpu["uuid"], []).append(value)
    gpu_stats = {uuid: {key: stats([gpu.get(key) for gpu in values]) for key in (
        "memory_used_mib", "memory_total_mib", "memory_capacity_used_percent", "gpu_utilization_percent",
        "memory_utilization_percent", "power_w", "temperature_c")} for uuid, values in sorted(gpu_samples.items())}
    for uuid, values in gpu_samples.items():
        gpu_stats[uuid]["device"] = {key: values[0].get(key) for key in ("uuid", "index", "name")}
    host_stats = {key: stats([row["host"].get(key) for row in selected]) for key in (
        "cpu_percent", "memory_available_bytes", "swap_used_bytes")}
    process_rss, process_cpu, matching_samples = [], [], 0
    for row in selected:
        matching = [p for p in row["processes"].get("rows", [])
                    if not roles or set(p["roles"]) & set(roles)]
        matching_samples += bool(matching)
        if roles:
            process_rss.append(sum(p["rss_bytes"] for p in matching) if matching else None)
        else:
            process_rss.append(row["processes"]["rss_bytes"])
        cpu = [p["cpu_percent"] for p in matching if number(p.get("cpu_percent")) is not None]
        process_cpu.append(sum(cpu) if cpu else None)
    return {"schema_version": 1, "total_samples": len(records), "selected_samples": len(selected),
            "filters": {"phases": list(phases), "min_transitions": min_transitions,
                        "max_gap_seconds": max_gap_seconds, "start_unix": start_unix,
                        "end_unix": end_unix, "gpus": list(gpu_selectors), "process_roles": list(roles)},
            "excluded": excluded,
            "intervals": interval_reports, "gpu": gpu_stats, "host": host_stats,
            "process_rss_bytes": stats(process_rss), "process_cpu_percent": stats(process_cpu),
            "matching_process_samples": matching_samples,
            "scope": {"gpu": "selected devices" if gpu_selectors else "all devices, including unrelated users",
                      "processes": "union of selected roles" if roles else "all observed processes, including unrelated GPU users",
                      "host": "whole host; unrelated users cannot be subtracted from host-level counters",
                      "memory_capacity_used_percent": "100 * memory_used_mib / memory_total_mib (VRAM capacity)",
                      "memory_utilization_percent": "nvidia-smi memory-controller busy percentage, not VRAM capacity"},
            "limitations": ["Sampled extrema; peaks shorter than the interval may be missed.",
                            "GPU/host totals include unrelated users; RSS sums include shared pages.",
                            "Status counters can lag: inspect status age and trainer/observed elapsed separately.",
                            "Only explicit driver phase markers can exclude checkpoint IO; status.state alone cannot.",
                            "Do not combine files from multiple hosts into a throughput timebase."]}


def role_value(value):
    if "=" not in value or not all(value.split("=", 1)):
        raise argparse.ArgumentTypeError("expected role=value")
    return tuple(value.split("=", 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    monitor = commands.add_parser("collect", help="read-only resource samples")
    monitor.add_argument("--output", type=Path, required=True)
    monitor.add_argument("--interval", type=float, default=2.)
    monitor.add_argument("--duration", type=float, default=1800.)
    monitor.add_argument("--query-timeout", type=float, default=2.)
    monitor.add_argument("--status", type=Path)
    monitor.add_argument("--phase-file", type=Path)
    monitor.add_argument("--pid", type=role_value, action="append", default=[])
    monitor.add_argument("--pid-file", type=role_value, action="append", default=[])
    report = commands.add_parser("summarize", help="summarize selected contiguous windows")
    report.add_argument("input", type=Path)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--phase", action="append", default=[])
    report.add_argument("--min-transitions", type=int, default=0)
    report.add_argument("--max-gap-seconds", type=float, default=5.)
    report.add_argument("--start-unix", type=float, help="inclusive wall-clock start, useful on remote host")
    report.add_argument("--end-unix", type=float, help="inclusive wall-clock end")
    report.add_argument("--gpu", action="append", default=[], help="GPU UUID or host physical index; repeatable")
    report.add_argument("--role", action="append", default=[], help="only these PID roles in process RSS/CPU; repeatable")
    args = parser.parse_args()
    if args.command == "collect":
        if not all(math.isfinite(v) and v > 0 for v in (args.interval, args.duration, args.query_timeout)):
            parser.error("interval, duration and query timeout must be finite and positive")
        try:
            args.pid = [(role, int(pid)) for role, pid in args.pid]
            if any(pid <= 0 for role, pid in args.pid):
                raise ValueError()
        except ValueError:
            parser.error("--pid values must be positive integers")
        roles = [role for role, _ in args.pid + args.pid_file]
        if len(roles) != len(set(roles)):
            parser.error("PID roles must be unique")
        collect(args)
    else:
        if args.min_transitions < 0 or not math.isfinite(args.max_gap_seconds) or args.max_gap_seconds <= 0:
            parser.error("min transitions must be nonnegative; max gap must be finite and positive")
        if any(v is not None and not math.isfinite(v) for v in (args.start_unix, args.end_unix)):
            parser.error("time bounds must be finite")
        if args.start_unix is not None and args.end_unix is not None and args.end_unix < args.start_unix:
            parser.error("end time must not precede start time")
        raw = args.input.read_bytes()
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if any(row.get("schema_version") != 1 for row in records):
            parser.error("unsupported resource schema")
        if len({row["hostname"] for row in records}) > 1:
            parser.error("summarize one host file at a time")
        result = summarize(records, args.phase, args.min_transitions, args.max_gap_seconds,
                           args.start_unix, args.end_unix, args.gpu, args.role)
        result["input"] = {"path": str(args.input), "sha256": hashlib.sha256(raw).hexdigest()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"output": str(args.output), "selected_samples": result["selected_samples"]}))


if __name__ == "__main__":
    main()
