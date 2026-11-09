"""CPU-only tests of monitoring evidence, never start CUDA or services."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

import psutil
import pytest


SPEC = importlib.util.spec_from_file_location(
    "monitor_runtime_resources", Path(__file__).parents[1] / "scripts/monitor_runtime_resources.py")
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


def test_gpu_na_csv_and_compute_memory(monkeypatch):
    outputs = iter([
        '0, GPU-example, "Example, GPU", 121, 32768, 54, [N/A], [Not Supported], 40\n',
        'GPU-example, 123, [N/A]\n',
    ])
    monkeypatch.setattr(monitor.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, next(outputs), ""))
    result = monitor.gpu_snapshot()
    assert not result["errors"]
    assert result["devices"][0]["name"] == "Example, GPU"
    assert result["devices"][0]["memory_used_mib"] == 121
    assert result["devices"][0]["power_w"] is None
    assert result["devices"][0]["memory_utilization_percent"] is None
    assert result["processes"] == [{"gpu_uuid": "GPU-example", "pid": 123, "memory_used_mib": None}]


def test_gpu_failure_does_not_hide_second_query(monkeypatch):
    def run(*args, **kwargs):
        if "--query-gpu=" in args[0][1]:
            raise subprocess.TimeoutExpired(args[0], 2)
        return subprocess.CompletedProcess(args, 0, "GPU-ok, 9, 5\n", "")
    monkeypatch.setattr(monitor.subprocess, "run", run)
    result = monitor.gpu_snapshot()
    assert len(result["errors"]) == 1
    assert result["processes"][0]["pid"] == 9


def test_phase_priority_and_unreadable_marker(tmp_path):
    status = {"data": {"state": "updating", "phase": "update"}}
    assert monitor.phase_for(status, None)["source"] == "status.phase"
    assert monitor.phase_for(status, {"data": {"phase": "checkpoint"}})["name"] == "checkpoint"
    missing = monitor.read_json_file(tmp_path / "missing.json", time.time())
    assert monitor.phase_for(status, missing) == {
        "name": "marker_unavailable", "source": "marker", "explicit": False}
    assert not monitor.phase_for({"data": {"state": "running"}}, None)["explicit"]
    assert monitor.phase_for(status, {"data": {}})["name"] == "marker_unavailable"


def test_read_json_corrupt_then_atomic_replacement(tmp_path):
    path = tmp_path / "status.json"
    path.write_text("{")
    assert monitor.read_json_file(path, time.time())["data"] is None
    other = tmp_path / "new.json"
    other.write_text('{"transitions": 64}')
    other.replace(path)
    result = monitor.read_json_file(path, time.time())
    assert result["data"]["transitions"] == 64
    assert 0 <= result["age_seconds"] < 5


def test_process_sampler_first_cpu_null_and_identity(tmp_path):
    pid = os.getpid()
    sampler = monitor.ProcessSampler([("learner", pid)])
    first = sampler.sample([], time.monotonic())
    own = next(row for row in first["rows"] if row["pid"] == pid)
    assert own["cpu_percent"] is None
    assert own["rss_bytes"] > 0
    second = sampler.sample([], time.monotonic() + 1)
    assert next(row for row in second["rows"] if row["pid"] == pid)["cpu_percent"] >= 0
    assert second["owners"]["learner"]["alive"]
    sampler.bound["learner"] = (pid, psutil.Process(pid).create_time() - 1)
    assert not sampler.sample([], time.monotonic())["owners"]["learner"]["alive"]


def test_receipt_identity_is_verified_before_attaching(tmp_path):
    path = tmp_path / "receipt.json"
    sampler = monitor.ProcessSampler(pid_files=[("learner", path)])
    assert not sampler.sample([], time.monotonic())["owners"]["learner"]["alive"]
    path.write_text(json.dumps({"pid": os.getpid(), "start_ticks": "wrong"}))
    result = sampler.sample([], time.monotonic())
    assert not result["rows"]
    assert not result["owners"]["learner"]["alive"]
    path.write_text(json.dumps({"pid": os.getpid(), "start_ticks": monitor.start_ticks(os.getpid())}))
    assert sampler.sample([], time.monotonic())["owners"]["learner"]["alive"]


@pytest.mark.parametrize("receipt", [None, [], 1, {"pid": os.getpid(), "create_time": []}])
def test_invalid_receipt_is_recorded_not_fatal(tmp_path, receipt):
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    sampler = monitor.ProcessSampler(pid_files=[("learner", path)])
    result = sampler.sample([], time.monotonic())
    assert not result["owners"]["learner"]["alive"]
    assert not result["rows"]


def sample(t, phase="steady", transitions=None, run_id="run1", elapsed=None, explicit=True):
    return {
        "hostname": "host", "monotonic_seconds": float(t), "wall_time": 1000 + t,
        "phase": {"name": phase, "source": "marker", "explicit": explicit},
        "status": {"data": {"run_id": run_id, "transitions": t * 10 if transitions is None else transitions,
                            "elapsed_seconds": float(t if elapsed is None else elapsed)}, "age_seconds": .1},
        "gpu": {"devices": [{"uuid": "GPU-a", "memory_used_mib": 100 + t,
                              "gpu_utilization_percent": 60}]},
        "host": {"memory_available_bytes": 1000 - t}, "processes": {"rss_bytes": 50},
    }


def test_summary_excludes_startup_checkpoint_and_does_not_bridge():
    rows = [sample(0, "startup"), sample(1), sample(2), sample(3, "checkpoint"), sample(4), sample(5)]
    result = monitor.summarize(rows, phases=["steady"])
    assert result["selected_samples"] == 4
    assert result["excluded"] == {"phase_filter": 2}
    assert len(result["intervals"]) == 2
    assert [w["transition_delta"] for w in result["intervals"]] == [10, 10]
    assert [w["observed_seconds"] for w in result["intervals"]] == [1, 1]
    assert result["gpu"]["GPU-a"]["memory_used_mib"]["max"] == 105


@pytest.mark.parametrize("last", [sample(8), sample(2, run_id="run2"),
                                 sample(2, transitions=0), sample(2, elapsed=0)])
def test_summary_never_bridges_gaps_restarts_or_counter_reset(last):
    result = monitor.summarize([sample(0), sample(1), last], max_gap_seconds=3)
    assert [w["samples"] for w in result["intervals"]] == [2, 1]
    assert result["intervals"][1]["observed_transitions_per_second"] is None


def test_summary_warmup_and_stale_status_distinguish_timebases():
    rows = [sample(0), sample(1), sample(2, elapsed=1.5)]
    result = monitor.summarize(rows, min_transitions=10)
    interval = result["intervals"][0]
    assert result["selected_samples"] == 2
    assert interval["observed_transitions_per_second"] == 10
    assert interval["trainer_transitions_per_second"] == 20
    assert interval["max_status_age_seconds"] == .1


def test_summary_no_status_has_no_throughput_claim():
    rows = [sample(0), sample(1)]
    for row in rows:
        row["status"] = None
    interval = monitor.summarize(rows)["intervals"][0]
    assert interval["transition_delta"] is None
    assert interval["trainer_transitions_per_second"] is None


def test_empty_summary_is_serializable():
    result = monitor.summarize([])
    assert result["selected_samples"] == 0
    json.dumps(result, allow_nan=False)


def test_absolute_time_filter_and_counter_deltas_exclude_startup():
    rows = [sample(t) for t in range(4)]
    for index, row in enumerate(rows):
        row["host"]["swap_out_bytes"] = [0, 2000, 2010, 2011][index]
        row["host"]["network_counters"] = {"bytes_sent": index * 40}
    result = monitor.summarize(rows, start_unix=1001, end_unix=1002)
    assert result["selected_samples"] == 2
    assert result["excluded"] == {"time_window": 2}
    interval = result["intervals"][0]
    assert interval["host_counter_deltas"]["swap_out_bytes"] == 10
    assert interval["host_counter_deltas"]["network_counters.bytes_sent"] == 40
    assert interval["host_counter_deltas"]["disk_counters.read_bytes"] is None


def test_phase_reentry_between_samples_is_not_a_continuous_steady_window():
    rows = [sample(0), sample(1), sample(2), sample(3)]
    for i, row in enumerate(rows):
        row["marker"] = {"mtime_unix": 999 if i < 2 else 1001.8, "data": {"phase": "steady"}}
    result = monitor.summarize(rows, phases=["steady"])
    assert [w["samples"] for w in result["intervals"]] == [2, 2]
    assert sum(w["transition_delta"] for w in result["intervals"]) == 20


def test_implicit_steady_phase_is_not_accepted_as_explicit_steady():
    result = monitor.summarize([sample(0, explicit=False)], phases=["steady"])
    assert result["selected_samples"] == 0
    assert result["excluded"] == {"implicit_steady_phase": 1}


@pytest.mark.parametrize("key", ["transitions", "elapsed_seconds"])
def test_null_counter_breaks_window_without_crashing(key):
    rows = [sample(t) for t in range(5)]
    rows[2]["status"]["data"][key] = None
    result = monitor.summarize(rows)
    assert [w["samples"] for w in result["intervals"]] == [2, 1, 2]
    assert monitor.summarize(rows, min_transitions=5)["selected_samples"] >= 3


def test_gpu_and_role_filter_excludes_unrelated_users_and_deduplicates_roles():
    row = sample(0)
    row["gpu"]["devices"] = [
        {"uuid": "GPU-other", "index": 0, "memory_used_mib": 900, "memory_total_mib": 1000},
        {"uuid": "GPU-ours", "index": 1, "memory_used_mib": 400, "memory_total_mib": 1000,
         "memory_utilization_percent": 7},
    ]
    row["processes"] = {"rss_bytes": 1500, "rows": [
        {"roles": ["gpu_compute"], "rss_bytes": 1000, "cpu_percent": 90},
        {"roles": ["trainer", "inference", "gpu_compute"], "rss_bytes": 500, "cpu_percent": 20},
    ]}
    result = monitor.summarize([row], gpu_selectors=["1"], roles=["trainer", "inference"])
    assert list(result["gpu"]) == ["GPU-ours"]
    assert result["gpu"]["GPU-ours"]["memory_capacity_used_percent"]["max"] == 40
    assert result["gpu"]["GPU-ours"]["memory_utilization_percent"]["max"] == 7
    assert result["process_rss_bytes"]["max"] == 500
    assert result["process_cpu_percent"]["max"] == 20
    assert list(monitor.summarize([row], gpu_selectors=["GPU-ours"])["gpu"]) == ["GPU-ours"]


def test_missing_selected_process_roles_are_unknown_not_zero():
    result = monitor.summarize([sample(0)], roles=["trainer"])
    assert result["process_rss_bytes"]["max"] is None
    assert result["matching_process_samples"] == 0
