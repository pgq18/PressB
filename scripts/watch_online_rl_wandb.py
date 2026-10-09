#!/usr/bin/env python3
"""Read existing experiment logs and stream metrics to W&B without touching training.

Uses its own process/environment. No model loading, simulator RPC, or checkpoint
upload. History uses recorded transition/update counters; live status has its
own axis so polling cannot move historical loss curves backwards.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pressb.online_rl.log_monitor import TrainingLogReader


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def process_identity(pid):
    directory = Path("/proc") / str(pid)
    stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=pid, starttime_ticks=int(stat[19]),
                argv=(directory / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0"))


def alive(record):
    try:
        return bool(record) and process_identity(record["pid"]) == record
    except (OSError, KeyError, ValueError):
        return False


def finite(value):
    return type(value) in (float, int, bool) and math.isfinite(value)


def history_payload(row):
    """Group scalars without inventing episode timestamps or interpolating losses."""
    payload = {"train/transitions": row["transitions"], "train/updates": row["updates"]}
    for key, value in row.items():
        if not finite(value):
            continue
        if key.startswith("recent_sr_") or key.startswith("recent_count_") or key == "success_rate":
            group = "rollout"
        elif key.startswith("episode_seconds") or key.startswith("termination_recent_"):
            group = "episode"
        elif key.startswith("floor_recent_"):
            group = "buttons_recent_1000"
        elif "loss" in key:
            group = "loss"
        elif key.startswith("q_"):
            group = "value"
        elif key in ("alpha", "actor_entropy", "normalized_action_norm", "reward", "discount") or "grad_norm" in key:
            group = "policy"
        elif "per_second" in key or key == "speed_window_seconds":
            group = "performance"
        elif key in ("training_transitions", "drain_transitions", "training_transition_budget", "episodes",
                     "successes", "wall_seconds", "active_envs", "replay_size", "runnable_update_budget"):
            group = "progress"
        else:
            continue
        payload[f"{group}/{key}"] = value
    times, counts = row.get("timing_seconds", {}), row.get("rpc_counts", {})
    for name, seconds in times.items():
        if finite(seconds):
            payload[f"timing/{name}_seconds"] = seconds
        if counts.get(name, 0) > 0 and finite(seconds):
            payload[f"latency/{name}_mean_ms"] = 1000 * seconds / counts[name]
    return payload


def live_payload(row):
    keys = ("transitions", "training_transitions", "updates", "episodes", "successes", "success_rate",
            "recent_sr_100", "recent_sr_1000", "recent_count_100", "recent_count_1000",
            "episode_seconds_recent_1000", "transitions_per_second", "updates_per_second",
            "transitions_per_second_recent_60s", "updates_per_second_recent_60s", "speed_window_seconds",
            "wall_seconds", "runnable_update_budget", "active_envs", "drain_transitions")
    result = {f"live/{key}": row[key] for key in keys if finite(row.get(key))}
    budget = row.get("training_transition_budget", 0)
    if budget:
        result["live/progress_percent"] = 100 * row.get("training_transitions", row["transitions"]) / budget
    return result


def gpu_metrics():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
    metrics = {}
    for line in result.stdout.splitlines():
        index, *values = [item.strip() for item in line.split(",")]
        for name, text in zip(("utilization_percent", "memory_used_mib", "memory_total_mib", "temperature_c", "power_w"), values):
            try:
                value = float(text)
                if finite(value):
                    metrics[f"hardware/gpu_{index}_{name}"] = value
            except ValueError:
                pass
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Experiment directory containing train/ and queue_status.json")
    parser.add_argument("--project", default="pressb-online-rl")
    parser.add_argument("--entity")
    parser.add_argument("--name", help="Display name for a new W&B run; existing registrations keep their name")
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be at least 1 second")
    output = args.output.resolve()
    config = read_json(output / "train_config.json")
    if not config:
        parser.error("Expected an existing train_config.json")
    monitor = output / "wandb_monitor"
    monitor.mkdir(exist_ok=True)
    lock = (monitor / "monitor.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(monitor / "process.json", process_identity(os.getpid()))
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)

    import wandb
    api = wandb.Api(timeout=30)
    entity = args.entity or api.default_entity
    registration_path = monitor / "run.json"
    registration = read_json(registration_path)
    resume_rows = 0
    if registration:
        if registration["entity"] != entity or registration["project"] != args.project:
            raise ValueError("Existing monitor belongs to a different W&B entity/project")
        # Trust the server's acknowledged prefix, not a local queued-log cursor.
        # A replay near a network interruption can duplicate a point but cannot
        # silently skip the unsent history prefix.
        if registration.get("url"):
            try:
                remote = api.run(f"{entity}/{args.project}/{registration['id']}")
                resume_rows = int(remote.summary.get("monitor/history_rows", 0))
            except wandb.errors.CommError:
                raise RuntimeError("Cannot verify existing W&B history; retry when connectivity returns") from None
    else:
        registration = dict(id=uuid.uuid4().hex[:8], entity=entity, project=args.project,
                            name=args.name or output.name)
        write_json(registration_path, registration)

    run = wandb.init(entity=entity, project=args.project, id=registration["id"], name=registration["name"],
        resume="allow", mode="online", dir=str(monitor),
        config={**config, "base_model_frozen": True, "noise_actor_frozen": True,
                "residual_initialization": "fresh", "monitor_interval_seconds": args.interval,
                "recent_sr_unit": "completed episodes, exact prefix at the recorded episode counter",
                "recent_sr_windows": [100, 1000], "rate_unit": "transitions per wall-clock second",
                "transition_unit": "one environment action chunk, up to 7 controls at 30 Hz",
                "episode_duration_unit": "simulated seconds for the entire episode",
                "training_sr_is_fixed_condition_evaluation": False},
        tags=["isaacsim", "frozen-noise", "action-residual", f"gamma-{config['single_gamma']}", "400k"],
        notes="Live read-only log monitor; original training began before W&B attachment. Historical curves backfilled. "
              "live/* polls status every 5s; rollout/loss curves follow original metrics every 256 optimizer updates. "
              "SR values are fractions. Windows use min(window, completed episodes). "
              "Timing counters overlap because learner updates run during RPC; do not sum them as a time breakdown.",
        settings=wandb.Settings(disable_git=True, disable_code=True, save_code=False, console="off",
                                x_disable_stats=True, x_disable_meta=True, init_timeout=60))
    registration.update(url=run.url, sdk_version=wandb.__version__)
    write_json(registration_path, registration)
    for axis in ("train/transitions", "train/updates", "live/transitions", "eval/episodes", "monitor/unix_time"):
        run.define_metric(axis, hidden=True)
    for group in ("rollout", "episode", "buttons_recent_1000", "performance", "progress", "timing", "latency"):
        run.define_metric(group + "/*", step_metric="train/transitions", step_sync=False)
    for group in ("loss", "value", "policy"):
        run.define_metric(group + "/*", step_metric="train/updates", step_sync=False)
    run.define_metric("live/*", step_metric="live/transitions", step_sync=False)
    run.define_metric("eval/*", step_metric="eval/episodes", step_sync=False)
    run.define_metric("hardware/*", step_metric="monitor/unix_time", step_sync=False)
    reader = TrainingLogReader(output / "train")
    eval_reader = TrainingLogReader(output / "eval")
    history_rows = 0
    last_live = None
    last_eval = None
    last_hardware = 0
    last_console = 0
    exit_code = 0
    state = {"state": "starting", "url": run.url}
    print(json.dumps({"event": "wandb_ready", **registration}), flush=True)
    try:
        while not stopping:
            started = time.monotonic()
            records, latest = reader.poll()
            for row in records:
                history_rows += 1
                if history_rows <= resume_rows:
                    continue
                run.log({**history_payload(row), "monitor/history_rows": history_rows})
            queue = read_json(output / "queue_status.json", {})
            phase = queue.get("stage", "unknown")
            run.summary.update({"experiment/state": queue.get("state", "unknown"), "experiment/phase": phase})
            if latest:
                key = (latest.get("transitions"), latest.get("updates"), latest.get("episodes"), latest.get("state"))
                if key != last_live:
                    run.log(live_payload(latest))
                    last_live = key
                run.summary.update({"experiment/training_state": latest.get("state"),
                                    "experiment/draining": latest.get("draining", False)})
            if (output / "eval/status.json").exists():
                _, evaluation = eval_reader.poll()
                if evaluation and evaluation.get("episodes") != last_eval:
                    payload = {f"eval/{key}": evaluation[key] for key in
                               ("episodes", "successes", "success_rate", "transitions", "wall_seconds")
                               if finite(evaluation.get(key))}
                    payload.update({f"eval/{key}": value for key, value in evaluation.items()
                                    if key.startswith("floor_recent_") and finite(value)})
                    run.log(payload)
                    last_eval = evaluation.get("episodes")
            now = time.time()
            if now - last_hardware >= 30:
                try:
                    run.log({**gpu_metrics(), "monitor/unix_time": now})
                except (OSError, subprocess.SubprocessError) as error:
                    print(json.dumps({"event": "gpu_monitor_unavailable", "error": str(error)}), flush=True)
                last_hardware = now
            status_path = output / "train/status.json"
            age = now - status_path.stat().st_mtime if status_path.exists() else None
            training_alive = alive(read_json(output / "train_process.json"))
            run.summary.update({"monitor/state": "running", "monitor/last_poll_unix_time": now,
                                "monitor/training_status_age_seconds": age,
                                "monitor/training_process_alive": training_alive})
            state = dict(state="running", url=run.url, history_rows=history_rows,
                         updated_at=datetime.now(timezone.utc).isoformat(), experiment_state=queue.get("state"),
                         phase=phase, training_process_alive=training_alive, training_status_age_seconds=age,
                         latest={key: latest.get(key) for key in ("training_transitions", "updates", "episodes",
                                 "recent_sr_100", "recent_sr_1000", "transitions_per_second_recent_60s")} if latest else None)
            write_json(monitor / "status.json", state)
            if now - last_console >= 30:
                print(json.dumps({"event": "monitor_progress", **state}), flush=True)
                last_console = now
            terminal = queue.get("state") in ("complete", "failed", "stopped")
            if terminal:
                # The supervisor writes terminal state only after writer exit;
                # poll once more so records appended during this iteration flush.
                records, latest = reader.poll()
                for row in records:
                    history_rows += 1
                    if history_rows > resume_rows:
                        run.log({**history_payload(row), "monitor/history_rows": history_rows})
                if latest:
                    run.log(live_payload(latest))
                run.summary["experiment/final_state"] = queue["state"]
                exit_code = 1 if queue["state"] == "failed" else 0
                state["state"] = "complete"
                break
            if phase == "train" and age is not None and age > 120 and not training_alive:
                raise RuntimeError("Training process exited and supervisor has not published a terminal/eval state")
            time.sleep(max(0, args.interval - (time.monotonic() - started)))
        if stopping:
            state["state"] = "stopped"
    except BaseException as error:
        state.update(state="failed", error=f"{type(error).__name__}: {error}")
        run.summary["monitor/error"] = state["error"]
        exit_code = 1
        raise
    finally:
        write_json(monitor / "status.json", state)
        run.summary["monitor/state"] = state["state"]
        run.finish(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
