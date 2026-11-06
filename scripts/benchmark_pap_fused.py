#!/usr/bin/env python3
"""Qualify fused PAP sampling with recorded grasp physics and native RGB pixels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rl_nodes/pap_vector_fast_simulation.json")
    parser.add_argument("--replay-episode", type=Path,
                        default=ROOT / "outputs/wallx_pap_eval_20261023/run/episodes/red_000")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    args.output.mkdir(parents=True)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    from benchmark_pap_parallel import load_replay
    from pressb.tasks.pap.vector import PAPVectorBackend
    from pressb.runtime.contract import decode_image
    from pressb.runtime.image_transport import resize_for_transport
    cfg = json.loads(args.config.read_text())["backend"]["config"]
    cfg.update(num_envs=args.num_envs, gpu_ids=[0], scene_config=str(args.replay_episode / "config.json"),
               smoothing={"mode": "none", "alpha": .8}, randomization={"position": "fixed", "colors": ["red"]},
               max_episode_seconds=20., log_dir=str(args.output / "worker"))
    report = {"config": cfg, "chunks": [], "source_episode": str(args.replay_episode)}
    def record():
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    backend = None
    try:
        backend = PAPVectorBackend(cfg)
        report["metadata"] = backend.describe()
        record()
        print(json.dumps({"event": "ready"}), flush=True)
        commands, options, replay_metadata = load_replay(args.replay_episode, backend.config["control_hz"])
        report["replay"] = replay_metadata
        ids = list(range(args.num_envs))
        before = backend.timings["captures"]
        initial = backend.reset(ids, 42, options)
        assert backend.timings["captures"]-before == cfg["reset_render_frames"]
        print(json.dumps({"event": "initial_reset", "captures": backend.timings["captures"]-before}), flush=True)
        # Read the already rendered native tiles: do not trigger another frame.
        tiles = backend.split_rgb(backend.annotator.get_data())
        checked = []
        for i in sorted(set([0, args.num_envs//2, args.num_envs-1])):
            for j, view in enumerate(("up", "wrist")):
                image = initial[i]["observation"]["images"][view]
                decoded = np.asarray(decode_image(image))
                expected = np.asarray(resize_for_transport(Image.fromarray(tiles[2*i+j]), cfg["image_preprocessing"]))
                assert np.array_equal(decoded, expected)
                assert image["preprocessing"]["original_size"] == cfg["render_resolution"]
                checked.append({"env_id": i, "camera": view, "shape": list(decoded.shape), "max_pixel_difference": 0})
        report["producer_pixel_audit"] = checked
        record()
        print(json.dumps({"event": "pixel_audit", "checked_images": len(checked)}), flush=True)
        del tiles
        generations = np.zeros(args.num_envs, dtype=int)
        original_outcomes = {}
        cursor = 0
        for chunk in range(24):
            indices = np.minimum(np.arange(cursor, cursor+32), len(commands)-1)
            items = [{"env_id": i, "actions": (commands[indices] if generations[i] == 0 else
                     np.repeat(backend.initial[None], 32, axis=0)).tolist()} for i in ids]
            old_ticks = backend.ticks.copy()
            old_clock, captures = backend.world.current_time, backend.timings["captures"]
            old_world_tick = backend.world.current_time_step_index
            started = time.perf_counter()
            response = backend.step_and_reset(items, reset_seed=42+chunk, reset_options=options, max_active_envs=args.num_envs)
            seconds = time.perf_counter()-started
            assert backend.timings["captures"]-captures == max(cfg["observation_render_frames"],
                cfg["reset_render_frames"] if response["reset_items"] else 0)
            done = []
            for row in response["items"]:
                i = row["env_id"]
                assert row["info"]["episode_physics_steps"] == old_ticks[i] + row["info"]["physics_steps"]
                if row["terminated"] or row["truncated"]:
                    done.append(i)
                    assert row["terminal_observation_omitted"] is True
                    assert row["observation"]["images"] == {}
                    if generations[i] == 0:
                        original_outcomes[i] = row["info"]
                    assert row["info"]["episode_physics_steps"] > 0
                else:
                    assert row["info"]["episode_physics_steps"] == backend.ticks[i]
                    assert row["observation"]["state"] == backend._states()[i].tolist()
            assert [row["env_id"] for row in response["reset_items"]] == done
            for row in response["reset_items"]:
                i = row["env_id"]
                generations[i] += 1
                assert backend.ticks[i] == 0
                assert row["observation"]["images"]
            max_ticks = max(row["info"]["physics_steps"] for row in response["items"])
            assert backend.world.current_time_step_index-old_world_tick == max_ticks
            # PhysX stores dt in float32; compare clock duration with tolerance,
            # and enforce the actual number of ticks exactly above. Backend
            # capture/reset already require the clock itself to remain exactly
            # unchanged before/after all rendering and indexed state writes.
            clock_error = backend.world.current_time-old_clock-max_ticks*backend.config["physics_dt"]
            assert np.isclose(backend.world.current_time-old_clock, max_ticks*backend.config["physics_dt"], rtol=1e-6, atol=1e-7)
            report["chunks"].append({"chunk": chunk, "seconds": seconds, "captures": backend.timings["captures"]-captures,
                "reset_count": len(done), "remaining_original": args.num_envs-len(original_outcomes),
                "clock_float_error_seconds": clock_error,
                "world_ticks": max_ticks, "terminal_and_reset_states_distinct": True, "survivor_state_unchanged_by_reset": True})
            record()
            print(json.dumps(report["chunks"][-1]), flush=True)
            cursor += 32
            if len(original_outcomes) == args.num_envs:
                break
        assert len(original_outcomes) == args.num_envs
        report["original_outcomes"] = original_outcomes
        report["expert_successes"] = sum(row["success"] for row in original_outcomes.values())
        assert report["expert_successes"] == args.num_envs
        # Separate terminal-only protocol probe. The diagnostic override makes
        # all newly reset slots timeout on the next tick; it is not part of the
        # expert replay or production task settings.
        original_max_ticks = backend.max_ticks
        backend.max_ticks = 1
        captures = backend.timings["captures"]
        response = backend.step_and_reset([{"env_id": i, "actions": [backend.initial.tolist()]} for i in ids],
                                         reset_seed=1, reset_options=options, max_active_envs=0)
        backend.max_ticks = original_max_ticks
        assert all(row["truncated"] for row in response["items"])
        assert response["reset_items"] == []
        assert backend.timings["captures"] == captures
        report["all_timeout_budget_zero_probe"] = {"diagnostic_max_ticks_override": 1, "restored_max_ticks": original_max_ticks,
                                                   "terminal_rows": len(response["items"]), "captures": 0, "resets": 0}
        report["timings"] = backend.timings
        report["state"] = "completed"
        record()
        print(json.dumps({"state": "completed", "expert_successes": report["expert_successes"]}), flush=True)
    except BaseException as error:
        import traceback
        report["state"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        record()
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    main()
