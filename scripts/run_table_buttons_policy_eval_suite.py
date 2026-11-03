#!/usr/bin/env python3
"""Run paired tabletop-button policy evaluations with bounded Isaac processes.

Each runtime owns one cell and resumes the same episode IDs after rotation or a
process failure. A physical policy failure counts as an evaluation outcome;
missing/corrupt episodes and process errors remain separately visible.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("near", "medium", "far")
COLORS = ("red", "yellow", "blue")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def summarize_numbers(values):
    values = sorted(float(value) for value in values)
    if not values:
        return dict(count=0, mean=None, p50=None, p95=None, maximum=None, total=0.)
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Timing measurements must be finite and nonnegative")
    def quantile(fraction):
        index = fraction * (len(values) - 1)
        low, high = math.floor(index), math.ceil(index)
        return values[low] * (high - index) + values[high] * (index - low) if low != high else values[low]
    return dict(count=len(values), mean=sum(values) / len(values), p50=quantile(.5), p95=quantile(.95),
                maximum=max(values), total=sum(values))


def validate_plan(plan, verify_assets=True):
    if plan.get("schema") != "pressb.table_buttons.policy_eval_plan.v1":
        raise ValueError("Unsupported policy evaluation plan")
    cells = plan["cells"]
    joint6_scope = plan.get("evaluation_scope", {}).get("kind") in (
        "joint6_single_model", "joint6_multitask_model")
    if joint6_scope:
        # Scoped joint6 runs preserve all original scenes for their declared tasks.
        from audit_table_buttons_policy_eval import audit_plan
        audit_plan(plan)
    elif len(cells) != 18 or len({cell["cell_id"] for cell in cells}) != 18:
        raise ValueError("Require18 unique single/multitask task cells")
    by_task, checked = {}, set()
    for cell in cells:
        profile, color = cell["spacing_profile"], cell["target_color"]
        if profile not in PROFILES or color not in COLORS or cell["instruction"] != f"Press the {color} button.":
            raise ValueError("Cell instruction or scene does not match its task")
        expected = (plan["evaluation_scope"]["model_id"] if joint6_scope else
                    f"{profile}_{color}" if cell["model_type"] == "single" else f"{profile}_multitask")
        if cell["model_id"] != expected:
            raise ValueError("Model selector differs from its task/spacing")
        episodes = cell["episodes"]
        if [episode["episode_index"] for episode in episodes] != list(range(len(episodes))):
            raise ValueError("Episode IDs must be contiguous and unique")
        if len({episode["seed"] for episode in episodes}) != len(episodes):
            raise ValueError("Duplicate layout seeds in one cell")
        key = profile, color
        signature = [(entry["episode_index"], entry["seed"], entry["policy_seed"],
                      entry["layout_sha256"], entry["effective_config_sha256"]) for entry in episodes]
        if key in by_task and signature != by_task[key]:
            raise ValueError("Single/multitask cells must share scene and policy RNG")
        by_task[key] = signature
        if verify_assets:
            for episode in episodes:
                for stem in ("layout", "effective_config", "preflight"):
                    path, expected_hash = episode[stem + "_file"], episode[stem + "_sha256"]
                    if path not in checked:
                        if sha256(path) != expected_hash:
                            raise ValueError(f"Prepared asset changed: {path}")
                        checked.add(path)
    if sum(len(cell["episodes"]) for cell in cells) != plan["planned_episodes"]:
        raise ValueError("Plan episode count disagrees with its cells")
    return cells


def scheduling_order(cells):
    """Interleave spacing servers so workers do not all start on one H200."""
    return sorted(cells, key=lambda cell: (
        0 if cell["model_type"] == "single" else 1,
        COLORS.index(cell["target_color"]), PROFILES.index(cell["spacing_profile"])))


class EpisodeIndex:
    def __init__(self, expected_collision_policy=None):
        self.cache = {}
        self.expected_collision_policy = expected_collision_policy

    def read_episode(self, directory, cell, episode):
        metadata_path, report_path = directory / "metadata.json", directory / "report.json"
        if not metadata_path.is_file() or not report_path.is_file():
            return None
        stamp = metadata_path.stat().st_mtime_ns, report_path.stat().st_mtime_ns
        key = str(directory)
        if key in self.cache and self.cache[key][0] == stamp:
            return self.cache[key][1]
        try:
            metadata, report = read_json(metadata_path), read_json(report_path)
            if self.expected_collision_policy is not None and any(
                    document.get("collision_policy") != self.expected_collision_policy for document in (metadata, report)):
                raise ValueError("Episode collision policy differs from the frozen formal plan")
            if metadata.get("evaluation_status") != "complete" or report.get("evaluation_status") != "complete":
                raise ValueError("Incomplete/infrastructure episode is not a physical task outcome")
            for field in ("cell_id", "model_id", "spacing_profile", "target_color"):
                if metadata.get(field) != cell[field]:
                    raise ValueError(f"Episode {field} differs from the frozen plan")
            if metadata.get("episode_index") != episode["episode_index"] or metadata.get("seed") != episode["seed"]:
                raise ValueError("Episode identity/layout seed differs from frozen plan")
            if metadata.get("policy_seed") != episode["policy_seed"]:
                raise ValueError("Policy RNG differs from frozen paired seed")
            if type(metadata.get("success")) is not bool or metadata["success"] != report.get("success"):
                raise ValueError("Episode success flags disagree or are missing")
            if metadata.get("stop_reason") != report.get("stop_reason"):
                raise ValueError("Episode stop reasons disagree")
            if not isinstance(metadata.get("model_sha256"), str) or len(metadata["model_sha256"]) != 64:
                raise ValueError("Episode lacks a complete model identity digest")
            for required in ("data.npz", "physics.npz", "policy_requests.jsonl", "control.jsonl", "contacts.json", "unexpected_contacts.json"):
                if not (directory / required).is_file():
                    raise ValueError(f"Completed episode lacks {required}")
            dual_policy = isinstance(metadata.get("collision_policy"), dict) and metadata["collision_policy"].get("version") == "table_buttons_dual_contact_v1"
            if dual_policy:
                for key in ("task_success", "strict_collector_success"):
                    if type(metadata.get(key)) is not bool or metadata[key] != report.get(key):
                        raise ValueError(f"Missing/inconsistent dual outcome metric: {key}")
                if metadata["task_success"] != metadata["success"]:
                    raise ValueError("Primary success alias differs from task_success")
                unexpected = read_json(directory / "unexpected_contacts.json")
                closure = read_json(directory / "finger_closure_contacts.json")
                blocking = read_json(directory / "blocking_contacts.json")
                if not all(isinstance(value, list) for value in (unexpected, closure, blocking)):
                    raise ValueError("Physical contact logs must be lists")
                strict_success = metadata["strict_collector_success"]
                if strict_success != (metadata["task_success"] and not unexpected):
                    raise ValueError("Original collector metric disagrees with full unexpected-contact log")
                if metadata["task_success"] and blocking:
                    raise ValueError("Task succeeded despite a blocking contact")
            else:
                strict_success, closure = metadata["success"], []
            duration = metadata.get("sim_duration_s", report.get("sim_duration_s", report.get("duration_s")))
            if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
                raise ValueError("Invalid simulation duration")
            requests = [json.loads(line) for line in (directory / "policy_requests.jsonl").read_text().splitlines() if line.strip()]
            latency = [row.get("latency_s", row.get("round_trip_seconds")) for row in requests
                       if "latency_s" in row or "round_trip_seconds" in row]
            # Missing timing is explicitly reported, never silently converted
            # to zero. The independent audit checks request/action semantics.
            summarize_numbers(latency)
            record = dict(valid=True, episode_index=episode["episode_index"], seed=episode["seed"],
                success=metadata["success"], stop_reason=metadata.get("stop_reason", report.get("stop_reason")),
                strict_collector_success=strict_success, finger_closure_events=len(closure),
                sim_duration_s=duration, request_latencies_s=latency, request_count=len(requests),
                requests_without_latency=len(requests) - len(latency),
                model_sha256=metadata.get("model_sha256", report.get("model_sha256")), directory=str(directory))
        except Exception as error:
            record = dict(valid=False, episode_index=episode["episode_index"], directory=str(directory), error=repr(error))
        self.cache[key] = stamp, record
        return record

    def summarize_cell(self, root, cell):
        records, missing = [], []
        for episode in cell["episodes"]:
            directory = Path(root) / "episodes" / f"episode_{episode['episode_index']:06d}"
            record = self.read_episode(directory, cell, episode)
            if record is None:
                missing.append(episode["episode_index"])
            else:
                records.append(record)
        valid = [record for record in records if record["valid"]]
        invalid = [record for record in records if not record["valid"]]
        successes, total = sum(record["success"] for record in valid), len(cell["episodes"])
        strict_successes = sum(record["strict_collector_success"] for record in valid)
        closure_episodes = sum(record["finger_closure_events"] > 0 for record in valid)
        reasons = dict(Counter(record["stop_reason"] for record in valid))
        latencies = [value for record in valid for value in record["request_latencies_s"]]
        hashes = sorted({record["model_sha256"] for record in valid if record["model_sha256"]})
        if len(hashes) > 1:
            invalid.append(dict(error="More than one checkpoint digest was used in one cell", digests=hashes))
        complete = len(valid) == total and not invalid
        return dict(cell_id=cell["cell_id"], model_id=cell["model_id"], model_type=cell["model_type"],
            spacing_profile=cell["spacing_profile"], target_color=cell["target_color"], planned=total,
            completed=len(valid), successes=successes, task_failures=len(valid) - successes,
            strict_collector_successes=strict_successes, strict_collector_failures=len(valid)-strict_successes,
            finger_closure_episodes=closure_episodes,
            finger_closure_events=sum(record["finger_closure_events"] for record in valid),
            missing_count=len(missing), missing_episode_ids=missing, invalid=invalid,
            complete=complete, success_rate=successes / total if complete else None,
            task_success_rate=successes / total if complete else None,
            strict_collector_success_rate=strict_successes / total if complete else None,
            strict_collector_success_rate_over_completed=strict_successes / len(valid) if valid else None,
            finger_closure_episode_rate=closure_episodes / total if complete else None,
            finger_closure_episode_rate_over_completed=closure_episodes / len(valid) if valid else None,
            success_rate_over_completed=successes / len(valid) if valid else None, stop_reasons=reasons,
            simulation_duration_s=summarize_numbers([record["sim_duration_s"] for record in valid]),
            inference_latency_s=summarize_numbers(latencies), model_sha256=hashes,
            request_count=sum(record["request_count"] for record in valid),
            requests_without_latency=sum(record["requests_without_latency"] for record in valid),
            interpretation="Physical outcomes only; missing/invalid infrastructure results do not count as policy failures")


def aggregate_cells(cells):
    result = {}
    for category, key in (("by_spacing", "spacing_profile"), ("by_color", "target_color"), ("by_model", "model_id")):
        groups = {}
        for row in cells.values():
            group = groups.setdefault(row[key], dict(planned=0, completed=0, successes=0, task_failures=0,
                strict_collector_successes=0, strict_collector_failures=0, finger_closure_episodes=0,
                finger_closure_events=0, missing_count=0, invalid_count=0, stop_reasons=Counter()))
            for field in ("planned", "completed", "successes", "task_failures", "missing_count",
                          "strict_collector_successes", "strict_collector_failures", "finger_closure_episodes", "finger_closure_events"):
                group[field] += row[field]
            group["invalid_count"] += len(row["invalid"])
            group["stop_reasons"].update(row["stop_reasons"])
        for group in groups.values():
            group["complete"] = group["completed"] == group["planned"] and not group["invalid_count"]
            group["success_rate"] = group["successes"] / group["planned"] if group["complete"] else None
            group["task_success_rate"] = group["success_rate"]
            group["strict_collector_success_rate"] = group["strict_collector_successes"] / group["planned"] if group["complete"] else None
            group["strict_collector_success_rate_over_completed"] = group["strict_collector_successes"] / group["completed"] if group["completed"] else None
            group["finger_closure_episode_rate"] = group["finger_closure_episodes"] / group["planned"] if group["complete"] else None
            group["finger_closure_episode_rate_over_completed"] = group["finger_closure_episodes"] / group["completed"] if group["completed"] else None
            group["success_rate_over_completed"] = group["successes"] / group["completed"] if group["completed"] else None
            group["stop_reasons"] = dict(group["stop_reasons"])
        result[category] = groups
    return result


class Controller:
    def __init__(self, args):
        self.args, self.started = args, time.monotonic()
        self.plan = read_json(args.plan)
        all_cells = validate_plan(self.plan)
        unknown = set(args.cells or ()) - {cell["cell_id"] for cell in all_cells}
        if unknown:
            raise ValueError(f"Unknown cells: {sorted(unknown)}")
        self.cells = scheduling_order([cell for cell in all_cells if not args.cells or cell["cell_id"] in args.cells])
        self.status_path, self.summary_path = args.output / "suite_status.json", args.output / "summary.json"
        self.lock = threading.Lock()
        self.runtime = (ROOT / "scripts/eval_table_buttons_policy.py").resolve()
        if not self.runtime.is_file():
            raise FileNotFoundError(f"Policy runtime is not installed: {self.runtime}")
        self.plan_hash = sha256(args.plan)
        if self.status_path.exists():
            previous = read_json(self.status_path)
            if previous.get("plan_sha256") != self.plan_hash:
                raise ValueError("Existing suite belongs to a different evaluation plan")
            for cell, row in previous.get("cells", {}).items():
                path = Path(f"/proc/{row.get('pid', 0)}/cmdline")
                if path.exists() and str(self.runtime).encode() in path.read_bytes():
                    raise RuntimeError(f"An earlier runtime still owns cell {cell}; refusing duplicate execution")
        self.status = dict(schema="pressb.table_buttons.policy_eval_suite.v1", state="starting",
            started_at=utc_now(), controller_pid=os.getpid(), plan=str(args.plan), plan_sha256=self.plan_hash,
            runtime=str(self.runtime), runtime_sha256=sha256(self.runtime), gpu=args.gpu, workers=args.workers,
            episodes_per_process=args.episodes_per_process, cells={cell["cell_id"]: dict(state="queued") for cell in self.cells},
            settings=dict(display=os.environ.get("DISPLAY"), xauthority=os.environ.get("XAUTHORITY"),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                rendering_gpu="Isaac physical GPU index; same unmasked enumeration as collection"))
        self.summaries = {}
        (args.output / "logs").mkdir(parents=True, exist_ok=True)

    def update(self, cell_id=None, summary=None, **fields):
        with self.lock:
            (self.status["cells"][cell_id] if cell_id else self.status).update(fields)
            if summary is not None:
                self.summaries[cell_id] = summary
            self.status.update(updated_at=utc_now(), elapsed_seconds=time.monotonic() - self.started,
                completed=sum(row["completed"] for row in self.summaries.values()),
                requested=sum(len(cell["episodes"]) for cell in self.cells))
            atomic_json(self.status_path, self.status)
            atomic_json(self.summary_path, dict(schema="pressb.table_buttons.policy_eval_summary.v1",
                plan_sha256=self.plan_hash, state=self.status["state"], requested=self.status["requested"],
                metrics=self.plan.get("metrics", dict(primary="success")),
                completed=self.status["completed"], cells=self.summaries, **aggregate_cells(self.summaries),
                note="Partial rates use completed valid outcomes; final rates require every planned episode. Process errors remain separate."))

    def command(self, cell, cell_output):
        return [str(self.args.python), str(self.runtime), "--plan", str(self.args.plan),
            "--cell", cell["cell_id"], "--output", str(cell_output),
            "--inference", cell["inference_endpoint"], "--gpu", str(self.args.gpu),
            "--resume", "--max-new-episodes", str(self.args.episodes_per_process),
            "--record-video-first", str(self.args.record_video_first)]

    def environment(self):
        environment = dict(os.environ)
        # Match the original collector's physical GPU enumeration. DISPLAY and
        # XAUTHORITY must survive even for headless RTX sensor rendering.
        if environment.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
            raise ValueError("Unset CUDA_VISIBLE_DEVICES for the original physical --gpu index convention")
        environment.pop("PYTHONPATH", None)
        environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           PYTHONUNBUFFERED="1", OMNI_KIT_ACCEPT_EULA="YES")
        return environment

    def worker(self, cell):
        cell_id = cell["cell_id"]
        directory = self.args.output / "cells" / cell_id
        index = EpisodeIndex(self.plan["defaults"].get("collision_policy"))
        recoveries, rotations = 0, 0
        process = None
        try:
            while True:
                summary = index.summarize_cell(directory, cell)
                if summary["invalid"]:
                    raise ValueError(f"Cell has invalid existing outcomes: {summary['invalid']}")
                if summary["complete"]:
                    self.update(cell_id, summary, state="complete", pid=None, recoveries=recoveries, rotations=rotations)
                    return
                command = self.command(cell, directory)
                log_path = self.args.output / "logs" / f"{cell_id}.log"
                before = summary["completed"]
                with log_path.open("a") as log:
                    log.write(json.dumps(dict(event="runtime_launch", command=command, utc=utc_now(),
                        completed=before, rotations=rotations, recoveries=recoveries)) + "\n")
                    log.flush()
                    process = subprocess.Popen(command, cwd=ROOT, env=self.environment(), stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    self.update(cell_id, summary, state="running", pid=process.pid, log=str(log_path),
                        command=command, rotations=rotations, recoveries=recoveries)
                    last_progress = time.monotonic()
                    last_count = before
                    timed_out = False
                    while process.poll() is None:
                        time.sleep(self.args.poll_seconds)
                        summary = index.summarize_cell(directory, cell)
                        runtime_path = directory / "status.json"
                        runtime_status = read_json(runtime_path) if runtime_path.is_file() else None
                        if summary["completed"] != last_count:
                            last_progress, last_count = time.monotonic(), summary["completed"]
                        self.update(cell_id, summary, runtime_status=runtime_status)
                        if time.monotonic() - last_progress > self.args.stall_timeout_s:
                            timed_out = True
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait(timeout=30)
                    returncode = process.returncode
                summary = index.summarize_cell(directory, cell)
                self.update(cell_id, summary, pid=None, last_returncode=returncode)
                if summary["complete"] and not summary["invalid"]:
                    self.update(cell_id, summary, state="complete", cleanup_returncode=returncode)
                    return
                runtime_path = directory / "status.json"
                runtime_status = read_json(runtime_path) if runtime_path.is_file() else {}
                planned_rotation = (runtime_status.get("state") == "restart_required"
                                    and summary["completed"] > before and returncode in (0, 75))
                if planned_rotation:
                    rotations += 1
                    continue
                recoveries += 1
                self.update(cell_id, summary, state="recovering", recoveries=recoveries,
                    last_infrastructure_error=dict(returncode=returncode, stalled=timed_out,
                        runtime_status=runtime_status, completed_before=before, completed_after=summary["completed"]))
                if recoveries > self.args.max_recoveries:
                    raise RuntimeError(f"Runtime exceeded {self.args.max_recoveries} recovery attempts; see {log_path}")
                time.sleep(self.args.poll_seconds)
        except Exception as error:
            # Only this controller's still-live child may be stopped here.
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            self.update(cell_id, state="failed", pid=None, error=repr(error))

    def run(self):
        self.environment()
        for cell in self.cells:
            summary = EpisodeIndex(self.plan["defaults"].get("collision_policy")).summarize_cell(self.args.output / "cells" / cell["cell_id"], cell)
            self.update(cell["cell_id"], summary)
        self.update(state="evaluating")
        with ThreadPoolExecutor(max_workers=self.args.workers) as executor:
            list(executor.map(self.worker, self.cells))
        success = all(row["state"] == "complete" for row in self.status["cells"].values())
        self.update(state="complete" if success else "failed", success=success, finished_at=utc_now())
        return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python", type=Path, default=ROOT / ".conda/envs/pressb/bin/python")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--episodes-per-process", type=int, default=20)
    parser.add_argument("--record-video-first", type=int, default=2)
    parser.add_argument("--max-recoveries", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=10.)
    parser.add_argument("--stall-timeout-s", type=float, default=600.)
    parser.add_argument("--cells", nargs="+")
    args = parser.parse_args()
    if not 1 <= args.workers <= 3 or not 1 <= args.episodes_per_process <= 50 or args.max_recoveries < 0:
        parser.error("Use1..3 workers,1..50 episodes per process and a nonnegative recovery budget")
    if args.poll_seconds <= 0 or args.stall_timeout_s <= 0 or args.record_video_first < 0:
        parser.error("Invalid polling, stall timeout or video quota")
    args.plan = args.plan.resolve()
    args.output = (args.output or args.plan.parent).resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "eval_suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return Controller(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
