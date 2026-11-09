#!/usr/bin/env python3
"""Exercise a native task's shared backend in real Isaac with >=2 environments.

No policy or learning is launched. Run once per task in a fresh simulator process.
Checks physical stepping, per-slot seeds, zero-tick reset isolation and RGB views.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("pap", "elevator", "table_buttons", "playground", "cube_stack", "tabletop_cleanup"))
    parser.add_argument("--config", type=Path, required=True, help="Raw task config or resolved simulation node JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=2)
    args = parser.parse_args()
    if args.num_envs < 2:
        parser.error("The isolation test requires at least two environments")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.config.read_text())
    config = deepcopy(data.get("backend", {}).get("config", data))
    config["num_envs"] = args.num_envs
    if args.task == "pap":
        config["gpu_ids"] = [args.gpu]
    else:
        config["gpu"] = args.gpu
    if args.task == "elevator":
        config["output"] = str(output/"native")
    else:
        config["log_dir"] = str(output/"workers")
    (output/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    from pressb.runtime.server import load_backend
    from pressb.runtime.parallel import SharedWorldBackend
    from pressb.runtime.contract import decode_image
    from pressb.control.pose_protocol import pose8_to_pose9
    backend = None
    result = {"task": args.task, "num_envs": args.num_envs, "gpu": args.gpu,
              "started_unix": time.time(), "physical_robot": False}
    try:
        backend = load_backend({"plugin": args.task, "config": config}, kind="task")
        assert isinstance(backend, SharedWorldBackend), "Default task did not instantiate the common backend"
        meta = backend.describe()
        (output/"metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
        ids = list(range(args.num_envs))
        seeds = {i: 20261109+37*i for i in ids}
        initial = backend.reset(ids, 20261109, {}, env_seeds=seeds)
        for row in initial:
            for name, encoded in row["observation"]["images"].items():
                decode_image(encoded).save(output/f"initial_env{row['env_id']}_{name}.png")
        def holds(rows):
            commands = []
            for row in rows:
                state = np.asarray(row["observation"]["state"])
                if meta["action"]["representation"] == "joint_position":
                    command = state[:len(meta["action"]["names"])]
                else:
                    command = pose8_to_pose9(state)
                commands.append({"env_id": row["env_id"], "actions": np.asarray([command]).tolist()})
            return commands
        before = backend.snapshot()
        transitioned = backend.step(holds(initial))
        assert all(row["info"]["physics_steps"] > 0 for row in transitioned)
        peer_ids = ids[1:]
        peers = backend.snapshot(peer_ids)
        if backend.ready[0]:
            backend.end_episode([0])
        reset = backend.reset([0], 101, {}, env_seeds={0: 8675309})
        assert backend.snapshot(peer_ids) == peers, "Indexed reset changed an untouched environment"
        for name, encoded in reset[0]["observation"]["images"].items():
            decode_image(encoded).save(output/f"reset_env0_{name}.png")
        first_spec = deepcopy(reset[0]["info"]["reset"])
        backend.end_episode([0])
        repeated = backend.reset([0], 777, {}, env_seeds={0: 8675309})
        assert repeated[0]["info"]["reset"] == first_spec, "Per-env reset seed changed with unrelated scalar seed"
        assert backend.snapshot(peer_ids) == peers
        live_rows = repeated+[row for row in transitioned if row["env_id"] in peer_ids and backend.ready[row["env_id"]]]
        second = backend.step(holds(live_rows))
        result.update(passed=True, shared_backend=type(backend).__name__, backend_module=type(backend).__module__,
            indexed_reset_zero_ticks=True, peer_snapshot_unchanged=True, independent_seed_repeatable=True,
            initial=initial, before_step=before, first_step=transitioned, reset=reset,
            second_step=second, final_snapshot=backend.snapshot(), timings=deepcopy(backend.timings))
    except BaseException as exc:
        import traceback
        result.update(passed=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        result["finished_unix"] = time.time()
        # RGB is already written as PNG, keep the audit JSON small.
        for key in ("initial", "first_step", "reset", "second_step"):
            for row in result.get(key, []):
                row["observation"]["images"] = {name: "saved_or_validated_RGB" for name in row["observation"]["images"]}
        (output/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        if backend is not None:
            backend.close()
    print(json.dumps({"passed": result["passed"], "task": args.task, "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
