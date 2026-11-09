#!/usr/bin/env python3
"""Exercise the shared parallel elevator task through the schema-v2 service.

Checks measured-state holds, exact timeout duration, indexed reset isolation,
and an optional recorded successful pose9 rollout. Does not contact a service.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_REPLAY = ROOT / "outputs/online_rl_combined_eval_20261006/combined_eval/policy_actions.jsonl"
DEFAULT_EPISODE = "e33c3e36482242ea981def5c20915e66:9"


def integer(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--gpu-id", type=integer, default=1)
    result.add_argument("--num-envs", type=int, default=2)
    result.add_argument("--max-seconds", type=float, default=2.)
    result.add_argument("--seed", type=integer, default=20260994)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--replay-log", type=Path, default=DEFAULT_REPLAY)
    result.add_argument("--replay-episode-id", default=DEFAULT_EPISODE)
    result.add_argument("--skip-replay", action="store_true", help="Run timeout/reset checks only")
    result.add_argument("--gpu-dynamics", action=argparse.BooleanOptionalAction, default=False,
        help="Shared backend currently requires --no-gpu-dynamics (CPU PhysX tensor control)")
    return result


def require(value, message):
    if not value:
        raise RuntimeError(message)


def load_replay(path, episode_id):
    """Read only the selected JSONL episode, keeping full seven-action chunks."""
    import numpy as np

    chunks = []
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            row = json.loads(raw)
            if row.get("episode_id") == episode_id:
                chunks.append(row)
    chunks.sort(key=lambda row: row["physics_index"])
    require(bool(chunks), "Replay episode was not found")
    require([row["physics_index"] for row in chunks] == list(range(0, len(chunks) * 28, 28)),
        "Replay chunks must be complete and start at physics tick zero")
    values = [row.get("combined_actions_pose9", row.get("actions_pose9")) for row in chunks]
    actions = np.asarray(values, dtype=float)
    require(actions.shape == (len(chunks), 7, 9) and np.isfinite(actions).all(),
        "Replay must contain finite seven-by-nine pose chunks")
    episodes = path.parent / "episodes.jsonl"
    match = None
    with episodes.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("episode_id") == episode_id:
                match = row
                break
    require(match is not None and match.get("success") is True, "Replay lacks a successful source outcome")
    layout = {key: match["layout"][key] for key in ("floor", "offset_x_m", "offset_y_m")}
    return actions.tolist(), layout, dict(log=str(path.resolve()), log_sha256=digest.hexdigest(),
        episode_id=episode_id, chunks=len(chunks), layout=layout,
        source_success=True, source_sim_seconds=match["sim_seconds"])


def raw_snapshot(backend, ids):
    """Read physical buffers directly without stepping or rendering."""
    engine = backend.engine
    return dict(world_time=float(engine.world.current_time),
        world_tick=int(engine.world.current_time_step_index),
        slots=backend.snapshot(ids))


def save_images(rows, directory):
    from pressb.runtime.contract import decode_image

    directory.mkdir(parents=True, exist_ok=False)
    saved = []
    for row in rows:
        images = row["observation"]["images"]
        require(set(images) == {"global", "wrist"}, "Elevator observation lacks both camera images")
        for camera, payload in images.items():
            frame = decode_image(payload)
            target = directory / f"env_{row['env_id']:03d}_{camera}.png"
            frame.save(target)
            saved.append(dict(env_id=row["env_id"], camera=camera, path=str(target), size=list(frame.size)))
    return saved


def run(args, *, backend_factory=None):
    import numpy as np
    from pressb.control.pose_protocol import pose8_to_pose9
    from pressb.runtime.simulation import SimulationService

    if args.output_dir is None:
        parent = ROOT / "outputs/elevator_task_smoke"
        parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="run_", dir=parent))
    else:
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
    report = dict(kind="shared_parallel_elevator_task_smoke", status="running", closed=False,
        close_attempted=False, output_dir=str(output), num_envs=args.num_envs, gpu_id=args.gpu_id,
        max_seconds=args.max_seconds, gpu_dynamics=args.gpu_dynamics,
        timing={}, images={}, stages=[], source_identity={})
    backend = service = None
    started = time.perf_counter()
    run_id = "elevator-smoke-" + uuid.uuid4().hex
    serial = 0

    def record():
        report["timing"]["wall_seconds"] = time.perf_counter() - started
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def request(**payload):
        nonlocal serial
        serial += 1
        return dict(run_id=run_id, request_id=str(serial), **payload)

    def reset(ids, seed, options):
        return service.reset(request(env_ids=ids, seed=seed, options=options))["items"]

    def step(rows, chunk=None):
        commands = []
        for row in rows:
            actions = chunk if chunk is not None else pose8_to_pose9(
                np.repeat(np.asarray(row["observation"]["state"])[None], 7, axis=0)).tolist()
            commands.append(dict(env_id=row["env_id"], episode_id=row["episode_id"],
                step_id=row["step_id"], actions=actions))
        return service.step(request(items=commands))["items"]

    def rollout(rows, name, chunks=None):
        results, elapsed, calls = {}, {row["env_id"]: 0 for row in rows}, []
        maximum = math.ceil(args.max_seconds * 120 / 28)
        for index in range(maximum):
            chunk = chunks[index] if chunks is not None and index < len(chunks) else None
            advanced = step(rows, chunk)
            calls.append({str(row["env_id"]): row["info"]["physics_steps"] for row in advanced})
            for row in advanced:
                elapsed[row["env_id"]] += row["info"]["physics_steps"]
                if row["terminated"] or row["truncated"]:
                    results[row["env_id"]] = row
            rows = [row for row in advanced if not (row["terminated"] or row["truncated"])]
            if not rows:
                break
        require(not rows, name + " did not terminate within the configured episode budget")
        terminal = [results[i] for i in sorted(results)]
        report["stages"].append(dict(name=name, chunks=len(calls), physics_steps_by_chunk=calls,
            episodes=[dict(env_id=row["env_id"], task=row["observation"]["task"],
                success=row["info"]["success"], terminated=row["terminated"], truncated=row["truncated"],
                physics_steps=elapsed[row["env_id"]], termination=row["info"].get("termination"))
                for row in terminal]))
        report["images"][name] = save_images(terminal, output / name)
        record()
        return terminal, elapsed

    try:
        chunks = layout = None
        if not args.skip_replay:
            chunks, layout, report["replay_source"] = load_replay(args.replay_log, args.replay_episode_id)
            require(args.max_seconds >= report["replay_source"]["source_sim_seconds"],
                "max-seconds is shorter than the recorded successful episode")
        config = dict(output=str(output / "simulation"), gpu=args.gpu_id, num_envs=args.num_envs,
            max_seconds=args.max_seconds, gpu_dynamics=args.gpu_dynamics, smoothing_window=3,
            single_gamma=.99, camera_resolution=224, render_subframes=1, light_settle_captures=1,
            light_settle_subframes=2, ik_iterations=16, log_actions=True, png_workers=4, cpu_threads=8)
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        if backend_factory is None:
            from pressb.tasks.elevator.native import NativeElevatorBackend
            backend_factory = NativeElevatorBackend
        before = time.perf_counter()
        backend = backend_factory(config)
        report["timing"]["startup_seconds"] = time.perf_counter() - before
        service = SimulationService(backend)
        metadata = service.health()
        require(metadata["task_id"] == "elevator" and metadata["num_envs"] == args.num_envs,
            "Unexpected elevator metadata")
        report["metadata"] = metadata
        report["source_identity"] = metadata.get("source_identity", {})
        report["legacy_identity"] = metadata.get("legacy_identity", {})
        ids = list(range(args.num_envs))
        initial = reset(ids, args.seed, {})
        report["images"]["initial"] = save_images(initial, output / "initial")
        terminal, elapsed = rollout(initial, "hold_timeout")
        require(all(row["truncated"] and not row["terminated"] for row in terminal),
            "Measured-state holds must end by timeout")
        require(all(value == round(args.max_seconds * 120) for value in elapsed.values()),
            "Hold timeout did not execute the exact configured physics duration")
        ready = []
        reset_checks = []
        for subset in ([0], ids[1:]):
            peers = [i for i in ids if i not in subset]
            before = raw_snapshot(backend, peers)
            peer_slots = {i: deepcopy(service.slots[i]) for i in peers}
            fresh = reset(subset, args.seed + 1, layout or {})
            after = raw_snapshot(backend, peers)
            require(after == before, "Indexed reset changed peer physical buffers or the world clock")
            require(all(service.slots[i] == peer_slots[i] for i in peers),
                "Indexed reset changed peer schema-v2 episode identity")
            ready.extend(fresh)
            reset_checks.append(dict(reset_env_ids=subset, peer_env_ids=peers,
                physical_state_unchanged=True, peer_episode_identity_unchanged=True,
                world_time_before=before["world_time"], world_time_after=after["world_time"]))
        ready.sort(key=lambda row: row["env_id"])
        report["indexed_reset"] = reset_checks
        report["images"]["indexed_reset"] = save_images(ready, output / "indexed_reset")
        final, _ = rollout(ready, "success_replay" if chunks is not None else "second_hold_timeout", chunks)
        if chunks is not None:
            require(all(row["info"]["success"] and row["terminated"] and not row["truncated"] for row in final),
                "Recorded replay did not physically press the target in every environment")
        report["close_run"] = service.close_run({"run_id": run_id})
        require(not report["close_run"]["restart_required"], "Completed episodes unexpectedly require restart")
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    except BaseException as error:
        report["status"] = "interrupted"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if backend is not None:
            report["timing"]["backend"] = dict(backend.timings)
            report["operations_status"] = report["status"]
            report["close_attempted"] = True
            if report["status"] == "passed":
                report["status"] = "closing"
            record()
            try:
                before = time.perf_counter()
                backend.close()
                report["timing"]["close_seconds"] = time.perf_counter() - before
                report["closed"] = True
                if report["status"] == "closing":
                    report["status"] = "passed"
            except Exception as error:
                report["status"] = "failed"
                report["close_error"] = f"{type(error).__name__}: {error}"
        record()
        print(json.dumps({key: report[key] for key in ("status", "closed", "output_dir", "timing", "source_identity")},
            separators=(",", ":"), allow_nan=False), flush=True)
    return 0 if report["status"] == "passed" else 1


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.num_envs < 2:
        arguments.error("--num-envs must be at least 2 for indexed reset isolation")
    if (not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 120
            or abs(args.max_seconds * 120 - round(args.max_seconds * 120)) > 1e-7):
        arguments.error("--max-seconds must be positive and lie on a 120 Hz physics tick")
    if args.output_dir is not None and args.output_dir.expanduser().exists():
        arguments.error("--output-dir must be new")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
