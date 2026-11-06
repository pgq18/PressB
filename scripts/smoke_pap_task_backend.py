#!/usr/bin/env python3
"""Smoke-test the direct PAP task backend with measured-state hold actions.

Runs describe, reset, one ordinary action, one fused timeout/reset action, and
close. The episode lasts two control periods; all other task/physics/rendering
settings come from the supplied configuration. No service or policy is used.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path,
        default=ROOT / "configs/rl_nodes/pap_vector_fast_simulation.json")
    result.add_argument("--num-envs", type=positive_int, default=2)
    result.add_argument("--gpu-id", type=nonnegative_int, default=1)
    result.add_argument("--seed", type=nonnegative_int, default=42)
    result.add_argument("--output-dir", type=Path,
        help="New directory; default creates a unique directory under outputs/pap_task_smoke")
    return result


def resolved_config(args, output_dir):
    config = json.loads(args.config.expanduser().resolve().read_text())["backend"]["config"]
    scene_path = Path(config.get("scene_config", "configs/piper_pick_place.json")).expanduser()
    if not scene_path.is_absolute():
        scene_path = ROOT / scene_path
    scene = json.loads(scene_path.read_text())
    scene.update(config.get("scene_overrides", {}))
    control_hz = float(scene["fps"])
    if not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("Scene fps must be positive and finite")
    config.update(num_envs=args.num_envs, gpu_ids=[args.gpu_id], gpu=args.gpu_id,
        python=sys.executable, max_episode_seconds=2. / control_hz,
        log_dir=str(output_dir / "workers"))
    config.pop("workers", None)
    return config


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def measured_state(row):
    state = row["observation"]["state"]
    require(isinstance(state, list) and len(state) == 7
        and all(type(value) in (int, float) and math.isfinite(value) for value in state),
        "PAP observation must contain seven finite joint/gripper values")
    return state


def check_rows(rows, ids, *, images=True):
    require(len(rows) == len(ids) and {row["env_id"] for row in rows} == set(ids),
        "PAP response omitted or duplicated an environment")
    for row in rows:
        measured_state(row)
        payloads = row["observation"]["images"]
        if images:
            require(set(payloads) == {"up", "wrist"}, "PAP observation is missing camera images")
            require(all(value.get("encoding") == "png_base64" and value.get("data")
                for value in payloads.values()), "PAP camera payload is empty or malformed")
        else:
            require(payloads == {}, "Fused terminal observations must omit images")


def hold_items(rows, metadata):
    lower, upper = metadata["action"]["lower"], metadata["action"]["upper"]
    require(len(lower) == len(upper) == 7, "PAP metadata must expose seven action bounds")
    # Clamp tiny settled-state excursions to the executable action limits.
    return [{"env_id": row["env_id"], "actions": [[
        min(max(value, lo), hi) for value, lo, hi in zip(measured_state(row), lower, upper)
    ]]} for row in rows]


def _output_directory(requested):
    if requested is not None:
        directory = requested.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=False)
        return directory
    parent = ROOT / "outputs" / "pap_task_smoke"
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run_", dir=parent))


def run(args, *, backend_factory=None):
    output = _output_directory(args.output_dir)
    report = dict(kind="pap_task_backend_smoke", status="running", output_dir=str(output),
        num_envs=args.num_envs, gpu_id=args.gpu_id, seed=args.seed, closed=False, close_attempted=False,
        hold_action_source="latest_observation.state_clipped_to_action_bounds",
        timing={}, source_identity={})
    backend = None
    started = time.perf_counter()

    def timed(name, function):
        before = time.perf_counter()
        try:
            return function()
        finally:
            report["timing"][name + "_seconds"] = time.perf_counter() - before

    def record():
        report["timing"]["wall_seconds"] = time.perf_counter() - started
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        config = resolved_config(args, output)
        (output / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
        report["max_episode_seconds"] = config["max_episode_seconds"]
        if backend_factory is None:
            from pressb.tasks.pap.vector import PAPVectorBackend
            backend_factory = PAPVectorBackend
        backend = timed("startup", lambda: backend_factory(config))
        metadata = timed("metadata", backend.describe)
        require(metadata["task_id"] == "pap" and metadata["num_envs"] == args.num_envs,
            "PAP metadata does not match the requested task/environment count")
        require("step_and_reset" in metadata["capabilities"], "PAP backend lacks fused reset support")
        report["source_identity"] = metadata["source_identity"]
        report["source_revision"] = metadata["source_revision"]
        report["execution"] = metadata["execution"]
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        ids = list(range(args.num_envs))
        ticks_before = backend.timings["world_ticks"]
        initial = timed("reset", lambda: backend.reset(ids, args.seed, {}))
        check_rows(initial, ids)
        require(backend.timings["world_ticks"] == ticks_before, "Initial indexed reset advanced physics")
        report["reset"] = dict(env_ids=ids, physics_steps=0, cameras_per_env=2)
        ordinary = timed("step", lambda: backend.step(hold_items(initial, metadata)))
        check_rows(ordinary, ids)
        require(all(not (row["terminated"] or row["truncated"]) for row in ordinary),
            "Hold episode ended before the fused timeout probe")
        report["step"] = dict(env_ids=ids,
            physics_steps_by_env={str(row["env_id"]): row["info"]["physics_steps"] for row in ordinary})
        before_fused = backend.timings["world_ticks"]
        fused = timed("fused", lambda: backend.step_and_reset(hold_items(ordinary, metadata),
            reset_seed=args.seed + 1, reset_options={}, max_active_envs=args.num_envs))
        terminal, reset = fused["items"], fused["reset_items"]
        check_rows(terminal, ids, images=False)
        check_rows(reset, ids)
        require(all(row["truncated"] and not row["terminated"]
            and row["info"]["termination_reason"] == "timeout"
            and row.get("terminal_observation_omitted") is True for row in terminal),
            "Fused step must preserve timeout transitions before resetting every environment")
        require(all(row["info"]["episode_physics_steps"] == 0 for row in reset),
            "Fused reset did not restart the episode clock")
        world_ticks = backend.timings["world_ticks"] - before_fused
        require(world_ticks == max(row["info"]["physics_steps"] for row in terminal),
            "Fused reset advanced additional physics ticks")
        report["fused"] = dict(timeout_env_ids=[row["env_id"] for row in terminal],
            reset_env_ids=[row["env_id"] for row in reset], world_ticks=world_ticks,
            terminal_images_omitted=True, reset_images_present=True, reset_episode_physics_steps=0)
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
            # Isaac's fast shutdown may exit the process inside close(). Keep
            # completed operation evidence even if Python never regains control.
            record()
            try:
                timed("close", backend.close)
                report["closed"] = True
                if report["status"] == "closing":
                    report["status"] = "passed"
            except Exception as error:
                report["status"] = "failed"
                report["close_error"] = f"{type(error).__name__}: {error}"
        record()
        print(json.dumps(report, separators=(",", ":"), allow_nan=False), flush=True)
    return 0 if report["status"] == "passed" else 1


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.output_dir is not None and args.output_dir.expanduser().exists():
        arguments.error("--output-dir must be a new directory")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
