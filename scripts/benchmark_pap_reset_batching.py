#!/usr/bin/env python3
"""Paired Isaac PAP reset benchmark through the production DPPO reset scheduler.

Uses one simulator and toggles only the advertised per-environment seed
capability. No policy is loaded, trained, or executed. Timings cover in-process
simulation service calls, not network traffic or end-to-end training.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    cfg = deepcopy(json.loads(args.config.read_text())["backend"]["config"])
    n = cfg["num_envs"]
    if args.repeats < 1 or any(k < 1 or k > n for k in args.counts):
        parser.error("invalid repeats or reset counts")
    args.output.mkdir(parents=True)
    cfg.pop("workers", None)
    cfg.update(gpu_ids=[args.gpu], log_dir=str(args.output / "worker"))
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    report = {"state": "starting", "config": cfg, "pairs": [],
        "scope": "production DPPO._reset and SimulationService on real Isaac; in-process service wall time; no model, RL, or network",
        "baseline": "same source with capability hidden, reproducing old singleton scheduling",
        "source_config": str(args.config.resolve()),
        "source_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def write():
        tmp = args.output / "report.tmp"
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        tmp.replace(args.output / "report.json")

    backend = None
    write()
    try:
        from pressb.tasks.pap.vector import PAPVectorBackend
        backend = PAPVectorBackend(cfg)
        from pressb.runtime.simulation import SimulationService
        from pressb.runtime.dppo_trainer import DPPOTrainer, DPPOTrainerConfig
        from pressb.runtime.contract import decode_image
        from pressb.runtime.image_transport import resize_for_transport

        service = SimulationService(backend, request_cache_size=2)
        report["metadata"] = service.health()
        ids = list(range(n))
        client_records = []

        class LocalClient:
            def call(self, path, payload=None):
                started = time.perf_counter()
                response = service.handlers[path](payload)
                seconds = time.perf_counter() - started
                # Pixel audit is deliberately outside the measured service call.
                if path == "/reset":
                    tiles = backend.engine.split_rgb(backend.engine.annotator.get_data())
                    checked = 0
                    for row in response["items"]:
                        eid = row["env_id"]
                        for j, view in enumerate(("up", "wrist")):
                            decoded = np.asarray(decode_image(row["observation"]["images"][view]))
                            preprocessing = backend.config["image_preprocessing"]
                            expected = (tiles[2*eid+j] if preprocessing is None else np.asarray(resize_for_transport(
                                Image.fromarray(tiles[2*eid+j]), preprocessing)))
                            assert np.array_equal(decoded, expected), (eid, view, "stale/wrong tile")
                            assert decoded.shape == expected.shape
                            assert tiles[2*eid+j].shape == (cfg["render_resolution"][1], cfg["render_resolution"][0], 3)
                            checked += 1
                    client_records.append({"seconds": seconds, "env_ids": payload["env_ids"],
                        "env_seeds": payload.get("env_seeds"), "images_checked": checked})
                return response

        client = LocalClient()
        trainer_dir = args.output / "trainer"
        trainer_dir.mkdir()
        trainer = DPPOTrainer(DPPOTrainerConfig(model={"factory": "unused"}, seed=43,
            output_dir=str(trainer_dir)), inference_client=object(), simulation_client=client)
        trainer.simulation_metadata = service.health()
        report["seed_schedule"] = trainer._seed_schedule()
        trainer._reset(ids, initial=True)
        shared_seed = trainer._seed(0, 0, "reset")
        assert all(backend.reset_specs[i]["seed"] == shared_seed for i in ids)
        assert len(client_records) == 1 and client_records[0]["env_seeds"] is None
        report["initial_shared_seed_preserved"] = True
        print(json.dumps({"event": "ready", "num_envs": n}), flush=True)

        def retire(targets):
            service.end_episode({"run_id": trainer.run_id, "request_id": uuid.uuid4().hex,
                "items": [{"env_id": i, "episode_id": service.slots[i]["episode_id"],
                           "step_id": service.slots[i]["step_id"]} for i in targets]})

        def prepare():
            retire(ids)
            response = service.reset({"run_id": trainer.run_id, "request_id": uuid.uuid4().hex,
                "env_ids": ids, "seed": 912, "options": {}})
            trainer.slots.update({row["env_id"]: row for row in response["items"]})
            # Same common layout, physical state and rendering warmup per branch.
            backend._observations(ids)

        permutation = np.random.default_rng(981).permutation(n).tolist()
        for count in args.counts:
            selected = permutation[:count]
            peers = [i for i in ids if i not in selected]
            for repeat in range(args.repeats):
                order = ["singleton", "batched"] if repeat % 2 == 0 else ["batched", "singleton"]
                pair = {"count": count, "repeat": repeat, "env_ids": selected, "order": order, "measurements": {}}
                snapshots, outputs, diagnostics, returned = {}, {}, {}, {}
                for mode in order:
                    prepare()
                    for i in ids:
                        trainer.episode_serial[i] = 3 + i % 7 + repeat * 11
                    trainer.simulation_metadata = service.health()
                    if mode == "singleton":
                        trainer.simulation_metadata["capabilities"].remove("per_env_reset_seeds")
                    retire(selected)
                    peer_before = backend.snapshot(peers) if peers else {}
                    peer_slots = {i: deepcopy(service.slots[i]) for i in peers}
                    clock = (backend.engine.world.current_time, backend.engine.world.current_time_step_index)
                    timing = deepcopy(backend.timings)
                    client_records.clear()
                    trainer._reset(selected)
                    measured = deepcopy(client_records)
                    assert (backend.engine.world.current_time, backend.engine.world.current_time_step_index) == clock
                    assert not peers or backend.snapshot(peers) == peer_before
                    assert {i: service.slots[i] for i in peers} == peer_slots
                    expected_calls = count if mode == "singleton" else 1
                    captures = backend.timings["captures"] - timing["captures"]
                    assert len(measured) == expected_calls
                    assert captures == expected_calls * cfg["reset_render_frames"]
                    assert all(trainer.slots[i]["step_id"] == 0 for i in selected)
                    expected_seeds = {i: trainer._seed(i, trainer.episode_serial[i], "reset") for i in selected}
                    assert all(backend.reset_specs[i]["seed"] == expected_seeds[i] == trainer.episode_seed[i]
                               for i in selected)
                    snapshots[mode] = backend.snapshot(selected)
                    outputs[mode] = {i: {"info": trainer.slots[i]["info"],
                        "observation": {key: value for key, value in trainer.slots[i]["observation"].items()
                                        if key != "images"}} for i in selected}
                    pair["measurements"][mode] = {"service_seconds": sum(r["seconds"] for r in measured),
                        "calls": len(measured), "captures": captures,
                        "render_seconds": backend.timings["render_seconds"] - timing["render_seconds"],
                        "encode_seconds": backend.timings["encode_seconds"] - timing["encode_seconds"],
                        "images_checked_against_live_tile": sum(r["images_checked"] for r in measured),
                        "peers_unchanged": True, "physics_ticks": 0, "requests": measured}
                    returned[mode] = {(i, view): np.asarray(decode_image(trainer.slots[i]["observation"]["images"][view]))
                                      for i in selected for view in ("up", "wrist")}
                    # Compare both branches under one common final scene, after the
                    # same diagnostic render boundary. Excluded from timings above.
                    obs = backend._observations(selected)
                    diagnostics[mode] = {(i, view): np.asarray(decode_image(obs[i]["images"][view]))
                                         for i in selected for view in ("up", "wrist")}
                    print(json.dumps({"event": "measurement", "count": count, "repeat": repeat,
                        "mode": mode, **{k: v for k, v in pair["measurements"][mode].items() if k != "requests"}}), flush=True)
                assert snapshots["singleton"] == snapshots["batched"]
                assert outputs["singleton"] == outputs["batched"]
                pair["physical_state_reset_spec_observation_state_equal"] = True
                pair["pixel_comparison"] = {}
                for label, images in (("returned", returned), ("common_final_scene", diagnostics)):
                    differences = []
                    for key in images["singleton"]:
                        diff = np.abs(images["singleton"][key].astype(np.int16) - images["batched"][key])
                        differences.append({"env_id": key[0], "camera": key[1], "mae": float(diff.mean()),
                            "p99": float(np.percentile(diff, 99)), "max": int(diff.max())})
                    pair["pixel_comparison"][label] = differences
                # Render history may change pixels, so report differences instead
                # of asserting byte equality across distinct rendering sequences.
                pair["speedup"] = (pair["measurements"]["singleton"]["service_seconds"] /
                                   pair["measurements"]["batched"]["service_seconds"])
                if count == 8 and repeat == 0:
                    previews = args.output / "previews"
                    previews.mkdir()
                    for mode in order:
                        for i in selected[:3]:
                            for view in ("up", "wrist"):
                                Image.fromarray(returned[mode][(i, view)]).save(previews / f"{mode}_{i}_{view}.png")
                report["pairs"].append(pair)
                write()
        report["state"] = "completed"
        write()
        print(json.dumps({"event": "completed", "pairs": len(report["pairs"])}), flush=True)
    except BaseException as error:
        import traceback
        report.update(state="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        write()
        raise
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    main()
