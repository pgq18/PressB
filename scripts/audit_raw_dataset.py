#!/usr/bin/env python3
"""Audit actual 120 Hz physics evidence and configured sample/action alignment."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pressb.panel_metadata import validate_episode_panel_metadata


def sampling_parameters(collection, metadata=None):
    """Validate recorded rates independently of the collector, including old 10 Hz captures."""
    schema = collection.get("raw_schema_version")
    if type(schema) is not int or schema not in (7, 9, 10, 11):
        raise ValueError(f"Unsupported raw schema: {schema!r}")
    fps = collection.get("fps")
    if type(fps) is not int or fps <= 0 or 120 % fps:
        raise ValueError("Sampling fps must be a positive integer divisor of 120 Hz")
    if schema < 10 and fps != 10:
        raise ValueError("Legacy raw schemas require their original 10 Hz sampling")
    stride = 120 // fps
    expected = {"fps": fps, "physics_hz": 120, "capture_stride": stride,
                "action_horizon_s": 1 / fps}
    physics_dt = collection.get("config", {}).get("physics_dt")
    if (isinstance(physics_dt, bool) or not isinstance(physics_dt, (int, float))
            or not np.isfinite(physics_dt) or abs(physics_dt - 1 / 120) > 1e-12):
        raise ValueError("Scene physics_dt must equal 1/120 second")
    for label, source in (("Collection", collection), ("Episode", metadata)):
        if source is None:
            continue
        if source.get("raw_schema_version") != schema:
            raise ValueError(f"{label} raw schema differs from collection")
        for key, value in expected.items():
            if key not in source and schema < 10 and key != "fps":
                continue
            actual = source.get(key)
            if key == "action_horizon_s":
                valid = (not isinstance(actual, bool) and isinstance(actual, (int, float))
                         and np.isfinite(actual) and abs(actual - value) <= 1e-12)
            else:
                valid = type(actual) is int and actual == value
            if not valid:
                raise ValueError(f"{label} {key} is inconsistent with {fps} Hz sampling")
        if (metadata is not None and "collection_fingerprint" in collection
                and source.get("collection_fingerprint") != collection["collection_fingerprint"]):
            raise ValueError(f"{label} fingerprint differs from collection")
    return expected


def audit(raw, episodes_per_task=100, allow_partial=False):
    report = dict(success=False, errors=[], episodes=[], floor_counts={}, total_frames=0)
    try:
        collection = json.loads((raw / "collection_metadata.json").read_text())
        rates = sampling_parameters(collection)
        config = collection["config"]
        home = np.asarray(config["home_q"])
        fps, stride = rates["fps"], rates["capture_stride"]
        report.update(rates)
    except Exception as exc:
        report["errors"].append(f"Collection: {type(exc).__name__}: {exc}")
        return report
    counts, identities, seeds, trajectories = Counter(), set(), set(), set()
    for directory in sorted(raw.glob("episode_*")):
        if not directory.is_dir() or not re.fullmatch(r"episode_[0-9]{6,}", directory.name):
            continue
        try:
            meta = json.loads((directory / "metadata.json").read_text())
            sampling_parameters(collection, meta)
            panel_offset_x, panel_offset_y = validate_episode_panel_metadata(collection, meta)
            with np.load(directory / "frames.npz", allow_pickle=False) as source:
                frames = dict(source)
            with np.load(directory / "physics.npz", allow_pickle=False) as source:
                physics = dict(source)
            floor, episode_id, seed = (meta[k] for k in ("floor", "episode_id", "seed"))
            assert meta["success"] is True and meta["task"] == f"Press {floor} floor."
            assert 24 <= floor <= 35 and directory.name == f"episode_{episode_id:06d}"
            assert episode_id not in identities and seed not in seeds
            identities.add(episode_id)
            seeds.add(seed)
            count = len(physics["q_actual"])
            assert count == meta["physics_steps"] and all(len(v) == count for v in physics.values())
            assert all(np.isfinite(v).all() for v in physics.values())
            assert all(physics[k].shape == (count, 12) for k in ("button_travel", "contact_force", "lights"))
            assert np.max(np.abs(physics["q_actual"][[0,-1]] - home)) < config["home_tolerance_rad"]
            assert np.max(np.abs(physics["q_actual"]-physics["q_command"])) < .15
            assert np.max(np.abs(physics["gripper_actual"]-[.004,-.004])) < .00025
            assert not meta["unexpected_collisions"] and meta["max_fk_error_m"] < .005
            assert np.max(physics["button_travel"][[0,-1]]) <= config["release_threshold"]
            assert not physics["lights"][[0,-1]].any()
            expected_lights = np.zeros(12, dtype=np.uint8)
            events = []
            for i, (travel, force, lights) in enumerate(zip(
                    physics["button_travel"], physics["contact_force"], physics["lights"])):
                for j in range(12):
                    if not expected_lights[j] and travel[j] >= config["press_threshold"] and force[j] > .02:
                        expected_lights[j] = 1
                        events.append(("pressed", j+24, i))
                    elif expected_lights[j] and travel[j] <= config["release_threshold"]:
                        expected_lights[j] = 0
                        events.append(("released", j+24, i))
                assert np.array_equal(lights, expected_lights), f"Unphysical lamp state at {i}"
            assert [e[:2] for e in events] == [("pressed",floor), ("released",floor)]
            assert events == [(e["type"], e["floor"], e["physics_index"]) for e in meta["events"]]
            indices = np.arange(0, count, stride)
            assert len(indices) == meta["num_frames"] == len(frames["state"])
            assert indices[-1] == count-1, "Last image must capture completed home"
            assert np.array_equal(frames["physics_index"], indices)
            assert np.allclose(frames["sim_time"], np.arange(len(indices)) / fps, atol=1e-10, rtol=0)
            original_length = meta["variation"]["validation"]["physics_samples_checked"]
            assert 1 <= original_length <= count and count - original_length < stride, "Invalid terminal padding"
            if "plan_physics_steps" in meta:
                assert meta["plan_physics_steps"] == original_length, "Plan length differs from variation metadata"
            assert np.isclose(meta["variation"]["physics_dt_s"], config["physics_dt"], atol=1e-12, rtol=0)
            assert all(np.isclose(event["time"], event["physics_index"] * config["physics_dt"],
                                  atol=1e-10, rtol=0) for event in meta["events"]), "Event time/physics index mismatch"
            command_hash = hashlib.sha256(np.ascontiguousarray(physics["q_command"][:original_length]).tobytes()).hexdigest()
            assert command_hash == meta["variation"]["joint_trajectory_sha256"]
            assert command_hash not in trajectories, "Duplicate commanded trajectory"
            trajectories.add(command_hash)
            future = np.minimum(indices+stride, original_length-1)
            assert np.array_equal(frames["q_actual"], physics["q_actual"][indices])
            assert np.array_equal(frames["q_target"], physics["q_command"][future]), "Action horizon mismatch"
            assert np.array_equal(frames["lights"], physics["lights"][indices])
            assert frames["lights"][:,floor-24].sum() > 0, "No image captures the illuminated target"
            assert np.allclose(frames["state"][:,7], np.diff(-physics["gripper_actual"][indices],axis=1)[:,0], atol=1e-10)
            assert np.allclose(frames["action"][:,7], .008, atol=1e-10)
            for key in ("state", "action"):
                assert frames[key].shape == (len(indices),8) and np.isfinite(frames[key]).all()
                assert np.allclose(np.linalg.norm(frames[key][:,3:7],axis=1),1,atol=1e-7)
            counts[floor] += 1
            report["total_frames"] += len(indices)
            report["episodes"].append(dict(episode_id=episode_id, floor=floor, frames=len(indices),
                panel_offset_x_m=panel_offset_x, panel_offset_y_m=panel_offset_y,
                maximum_button_travel_m=float(physics["button_travel"][:,floor-24].max()),
                maximum_contact_force_n=float(physics["contact_force"][:,floor-24].max()),
                final_home_error_rad=float(np.abs(physics["q_actual"][-1]-home).max())))
        except Exception as exc:
            report["errors"].append(f"{directory.name}: {type(exc).__name__}: {exc}")
    report["floor_counts"] = {str(f):counts[f] for f in range(24,36)}
    if not report["episodes"]:
        report["errors"].append("No valid episodes")
    if not allow_partial and any(counts[f] != episodes_per_task for f in range(24,36)):
        report["errors"].append(f"Expected {episodes_per_task} episodes for every floor")
    report["total_episodes"] = len(report["episodes"])
    report["success"] = not report["errors"]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = audit(args.raw, args.episodes_per_task, args.allow_partial)
    (args.raw / "physics_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != "episodes"}, indent=2))
    raise SystemExit(0 if result["success"] else 1)
