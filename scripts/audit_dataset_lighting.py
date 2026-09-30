#!/usr/bin/env python3
"""Check every committed episode's first RGB frame against a lit single-scene reference.

CPU only: decode exactly the first presentation frame of each camera video.
This verifies initial home-pose illumination, not every frame of the motion or
button-light transition timing. Existing dataset files are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import av
import numpy as np

from audit_raw_dataset import sampling_parameters


ROOT = Path(__file__).resolve().parents[1]
VIEWS = ("wrist", "global")
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
MAX_RELATIVE_LUMA_DIFFERENCE = 0.05
MAX_SATURATED_PIXEL_FRACTION = 0.01
SATURATED_CHANNEL_THRESHOLD = 250
ENVIRONMENT_SPACING_M = 6.0  # Production create_envs default; checked against each recorded offset.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def first_rgb(path, fps=None):
    """Close the decoder immediately after its first presentation frame."""
    with av.open(str(path)) as container:
        require(len(container.streams.video) == 1, "Expected exactly one video stream")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        if fps is not None:
            require(stream.average_rate is not None and abs(float(stream.average_rate) - fps) <= 1e-8,
                    f"Video frame rate differs from the collection's {fps} Hz")
        frame = next(container.decode(stream), None)
        require(frame is not None, "Video contains no decodable frame")
        require(frame.pts is not None and frame.time_base is not None, "First frame has no timestamp")
        timestamp = float(frame.pts * frame.time_base)
        require(abs(timestamp) < 1e-9, f"First video timestamp is not zero: {timestamp}")
        rgb = frame.to_ndarray(format="rgb24")
        require(rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8,
                f"Expected 640x480 RGB uint8, found {rgb.shape}/{rgb.dtype}")
        return rgb, timestamp


def image_metrics(rgb, reference_mean):
    luma = rgb @ LUMA_WEIGHTS
    mean = float(luma.mean())
    signed_difference = mean / reference_mean - 1.0
    saturated = float((rgb.max(axis=2) >= SATURATED_CHANNEL_THRESHOLD).mean())
    return {"mean_luma": mean, "reference_mean_luma": reference_mean,
            "signed_relative_luma_difference": signed_difference,
            "absolute_relative_luma_difference": abs(signed_difference),
            "relative_luma_difference_percent": 100.0 * signed_difference,
            "saturated_any_channel_fraction": saturated,
            "saturated_luma_fraction": float((luma >= SATURATED_CHANNEL_THRESHOLD).mean()),
            "dark_luma_fraction": float((luma < 10).mean())}


def reference_means(reference, collection):
    require(reference.get("success") is True, "Reference report did not pass")
    require(reference.get("num_envs") == 1, "Reference must contain exactly one environment")
    require(len(reference.get("active_domes", [])) == 1, "Reference must have one active Dome")
    require(reference["config"] == collection["config"], "Reference and dataset scene configurations differ")
    require(reference["snapshot_sha256"] == collection["scene_sha256"],
            "Reference and dataset scene snapshot hashes differ")
    home = np.asarray(collection["config"]["home_q"], dtype=np.float64)
    tolerance = float(collection["config"]["home_tolerance_rad"])
    means = {}
    for view in VIEWS:
        matches = [row for row in reference["images"]
                   if row["pose"] == "home" and row["camera"] == view and row["env_id"] == 0]
        require(len(matches) == 1, f"Expected one home reference for {view}")
        row = matches[0]
        q = np.asarray(row["joint_positions"], dtype=np.float64)[:6]
        require(q.shape == (6,) and np.isfinite(q).all() and np.max(np.abs(q - home)) < tolerance,
                f"Reference {view} is not in the configured home pose")
        mean = float(row["mean_luma"])
        require(np.isfinite(mean) and 0 < mean <= 255, f"Invalid {view} reference brightness")
        require(row["clipped_any_channel_fraction"] <= MAX_SATURATED_PIXEL_FRACTION,
                f"Reference {view} has excessive saturation")
        means[view] = mean
    return means


def initial_state(directory, metadata, config):
    home = np.asarray(config["home_q"], dtype=np.float64)
    tolerance = float(config["home_tolerance_rad"])
    require(home.shape == (6,) and np.isfinite(home).all() and tolerance > 0,
            "Invalid configured home pose/tolerance")
    recorded_error = float(metadata["initial_home_error_rad"])
    require(np.isfinite(recorded_error) and recorded_error < tolerance,
            "Metadata reports initial pose outside home tolerance")
    travel = np.asarray(metadata["initial_button_travel_m"], dtype=np.float64)
    require(travel.shape == (12,) and np.isfinite(travel).all()
            and np.max(np.abs(travel)) <= config["release_threshold"],
            "Metadata reports a button initially outside its released range")
    with np.load(directory / "frames.npz", allow_pickle=False) as frames:
        q = np.asarray(frames["q_actual"][0], dtype=np.float64)
        lights = frames["lights"][0]
        require(q.shape == (6,) and np.isfinite(q).all(), "Invalid first measured arm pose")
        error = float(np.max(np.abs(q - home)))
        require(error < tolerance, f"First measured pose is not home: {error} rad")
        require(lights.shape == (12,) and np.isfinite(lights).all() and not np.any(lights),
                "At least one recorded button light is on in the first sample")
        require(int(frames["physics_index"][0]) == 0 and float(frames["sim_time"][0]) == 0.0,
                "First sample must have physics_index=0 and sim_time=0")
        require(len(frames["q_actual"]) == metadata["num_frames"], "Frame count differs from metadata")
        phase = str(frames["phase"][0])
    return {"q_actual_rad": q.tolist(), "home_error_rad": error,
            "metadata_initial_home_error_rad": recorded_error,
            "all_recorded_lights_off": True, "phase": phase}


def environment_id(metadata):
    offset = np.asarray(metadata["env_offset_m"], dtype=np.float64)
    require(offset.shape == (3,) and np.isfinite(offset).all(), "Invalid environment offset")
    index = int(round(offset[1] / ENVIRONMENT_SPACING_M))
    require(index >= 0 and np.allclose(offset, [0., index * ENVIRONMENT_SPACING_M, 0.],
                                      rtol=0, atol=1e-7),
            "Environment offset does not follow the production 6 m Y-spacing convention")
    if "env_id" in metadata:
        require(metadata["env_id"] == index, "Explicit environment ID disagrees with offset")
    return index


def production_reference_means(reference_raw, collection):
    """Use an independently collected single-environment episode through the same capture chain."""
    manifest_path = reference_raw / "collection_metadata.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["raw_schema_version"] == collection["raw_schema_version"]
            and manifest["raw_schema_version"] in (9, 10),
            "Production reference mode requires matching schema 9 or 10 datasets")
    rates = sampling_parameters(collection)
    require(sampling_parameters(manifest) == rates,
            "Production reference and dataset sampling rates differ")
    require(manifest["collection_fingerprint"] == collection["collection_fingerprint"],
            "Production reference and dataset collection fingerprints differ")
    require(manifest["config"] == collection["config"], "Production reference scene configuration differs")
    statuses = sorted(reference_raw.glob("worker_*_status.json"))
    lighting_paths = sorted(reference_raw.glob("worker_*_lighting.json"))
    require(len(statuses) == len(lighting_paths) == 1,
            "Production reference must have one recorded single-environment worker")
    status = json.loads(statuses[0].read_text())
    require(status["num_envs"] == 1 and status["status"] == "complete",
            "Production reference worker must be complete with num_envs=1")
    lighting = json.loads(lighting_paths[0].read_text())
    require(lighting["collection_fingerprint"] == manifest["collection_fingerprint"],
            "Reference lighting evidence has a different fingerprint")
    require(len(lighting["domes"]) == 1 and lighting["domes"][0]["active"] is True
            and lighting["domes"][0]["path"] == "/World/envs/env_0/Lights/Ambient",
            "Production reference must contain only the active env_0 Dome")
    require(len(lighting["local_rect_lights"]) == 1 and len(lighting["local_rect_lights"][0]) == 2,
            "Production reference must record one room's two local RectLights")
    directory = reference_raw / "episode_000000"
    metadata = json.loads((directory / "metadata.json").read_text())
    require(metadata["success"] is True and metadata["episode_id"] == 0 and environment_id(metadata) == 0,
            "Production reference episode 0 must be successful and use env_0")
    require(metadata["collection_fingerprint"] == manifest["collection_fingerprint"]
            and metadata["raw_schema_version"] == manifest["raw_schema_version"],
            "Production reference episode identity differs")
    sampling_parameters(manifest, metadata)
    first_sample = initial_state(directory, metadata, manifest["config"])
    means, metrics = {}, {}
    for view in VIEWS:
        rgb, timestamp = first_rgb(directory / f"{view}.mp4", rates["fps"])
        mean = float((rgb @ LUMA_WEIGHTS).mean())
        require(mean > 0, f"Production reference {view} is black")
        metrics[view] = image_metrics(rgb, mean)
        metrics[view]["first_video_timestamp_s"] = timestamp
        require(metrics[view]["saturated_any_channel_fraction"] <= MAX_SATURATED_PIXEL_FRACTION,
                f"Production reference {view} is excessively saturated")
        means[view] = mean
    proof_paths = [manifest_path, statuses[0], lighting_paths[0], directory / "metadata.json",
                   directory / "frames.npz", directory / "wrist.mp4", directory / "global.mp4"]
    return means, {"identities": [identity(path) for path in proof_paths], "worker_status": status,
                   "first_sample": first_sample, "views": metrics,
                   "capture_method": manifest["capture_method"],
                   "scope": "Same production capture and MP4 chain; measures parallel vs single-environment "
                            "lighting. It does not replace the separately retained static-reference result."}


def audit(raw, reference_path=None, episodes_per_task=100, allow_partial=False, reference_raw=None):
    raw = Path(raw).resolve()
    reference_raw = Path(reference_raw).resolve() if reference_raw is not None else None
    reference_path = (Path(reference_path).resolve() if reference_path is not None else
                      ROOT / "outputs/lighting_fix_validation/single/report.json")
    report = {"success": False, "created_at": datetime.now(timezone.utc).isoformat(),
              "raw": str(raw), "reference": str(reference_raw or reference_path),
              "reference_mode": "single_environment_production" if reference_raw else "single_environment_static",
              "scope": "First decoded frame of both cameras for every committed episode; "
                       "verifies initial home illumination only, not all motion frames or light timing.",
              "expected_episodes_per_task": episodes_per_task, "allow_partial": allow_partial,
              "thresholds": {"absolute_relative_luma_difference": MAX_RELATIVE_LUMA_DIFFERENCE,
                             "saturated_any_channel_fraction": MAX_SATURATED_PIXEL_FRACTION,
                             "saturated_channel_threshold_uint8": SATURATED_CHANNEL_THRESHOLD,
                             "home_tolerance_rad": None},
              "luma_definition": "0.2126*R + 0.7152*G + 0.0722*B on decoded uint8 RGB (not physical radiance)",
              "environment_id_convention": "env_offset_m=[0,6*env_id,0], production clone spacing",
              "episodes": [], "env_groups": {}, "floor_counts": {}, "errors": []}
    try:
        require(episodes_per_task > 0, "episodes_per_task must be positive")
        collection_path = raw / "collection_metadata.json"
        collection = json.loads(collection_path.read_text())
        rates = sampling_parameters(collection)
        report.update(rates)
        if reference_raw is not None:
            means, proof = production_reference_means(reference_raw, collection)
            report["production_reference_evidence"] = proof
        else:
            reference = json.loads(reference_path.read_text())
            means = reference_means(reference, collection)
            report["reference_identity"] = identity(reference_path)
        config = collection["config"]
        report.update(collection_identity=identity(collection_path),
                      audit_script_identity=identity(Path(__file__)),
                      raw_schema_version=collection["raw_schema_version"], reference_means=means,
                      collection_fingerprint=collection["collection_fingerprint"])
        report["thresholds"]["home_tolerance_rad"] = config["home_tolerance_rad"]
    except Exception as exc:
        report["errors"].append(f"Inputs: {type(exc).__name__}: {exc}")
        return report

    directories = sorted(path for path in raw.glob("episode_*")
                         if path.is_dir() and re.fullmatch(r"episode_[0-9]{6,}", path.name))
    counts, groups, identities = Counter(), defaultdict(list), set()
    for directory in directories:
        row = {"directory": directory.name, "success": False, "views": {}, "errors": []}
        try:
            metadata = json.loads((directory / "metadata.json").read_text())
            sampling_parameters(collection, metadata)
            eid, floor = metadata["episode_id"], metadata["floor"]
            row.update(episode_id=eid, floor=floor)
            require(isinstance(eid, int) and eid >= 0 and eid not in identities,
                    "Invalid or duplicate episode ID")
            require(directory.name == f"episode_{eid:06d}", "Directory and episode ID differ")
            require(floor == 24 + eid % 12 and metadata["task"] == f"Press {floor} floor.",
                    "Episode floor or task text differs from source ID convention")
            require(metadata["success"] is True, "Episode metadata did not pass physical collection")
            require(metadata["raw_schema_version"] == collection["raw_schema_version"]
                    and metadata["collection_fingerprint"] == collection["collection_fingerprint"],
                    "Episode and collection identities differ")
            identities.add(eid)
            counts[floor] += 1
            row.update(env_id=environment_id(metadata), env_offset_m=metadata["env_offset_m"],
                       first_sample=initial_state(directory, metadata, config))
            for view in VIEWS:
                try:
                    rgb, timestamp = first_rgb(directory / f"{view}.mp4", rates["fps"])
                    metrics = image_metrics(rgb, means[view])
                    metrics.update(first_video_timestamp_s=timestamp, decoded_frames_checked=1,
                                   width=640, height=480, channels=3)
                    metrics["success"] = (metrics["absolute_relative_luma_difference"]
                                          <= MAX_RELATIVE_LUMA_DIFFERENCE
                                          and metrics["saturated_any_channel_fraction"]
                                          <= MAX_SATURATED_PIXEL_FRACTION)
                    row["views"][view] = metrics
                    groups[(row["env_id"], view)].append(metrics)
                    if metrics["absolute_relative_luma_difference"] > MAX_RELATIVE_LUMA_DIFFERENCE:
                        row["errors"].append(f"{view}: mean luma differs from reference by "
                            f"{metrics['relative_luma_difference_percent']:.3f}% (limit ±5%)")
                    if metrics["saturated_any_channel_fraction"] > MAX_SATURATED_PIXEL_FRACTION:
                        row["errors"].append(f"{view}: saturated pixel fraction "
                            f"{metrics['saturated_any_channel_fraction']:.6f} exceeds 0.01")
                except Exception as exc:
                    row["errors"].append(f"{view}: {type(exc).__name__}: {exc}")
        except Exception as exc:
            row["errors"].append(f"{type(exc).__name__}: {exc}")
        row["success"] = not row["errors"] and len(row["views"]) == 2
        report["episodes"].append(row)
        report["errors"].extend(f"{directory.name}: {message}" for message in row["errors"])

    for (env_id, view), values in sorted(groups.items()):
        report["env_groups"].setdefault(str(env_id), {})[view] = {
            "episodes_checked": len(values), "passed": sum(v["success"] for v in values),
            "mean_luma": float(np.mean([v["mean_luma"] for v in values])),
            "minimum_mean_luma": min(v["mean_luma"] for v in values),
            "maximum_mean_luma": max(v["mean_luma"] for v in values),
            "maximum_absolute_relative_luma_difference": max(v["absolute_relative_luma_difference"] for v in values),
            "maximum_saturated_any_channel_fraction": max(v["saturated_any_channel_fraction"] for v in values)}
    report.update(total_episodes=len(directories), passed_episodes=sum(row["success"] for row in report["episodes"]),
                  decoded_frames_checked=sum(len(row["views"]) for row in report["episodes"]),
                  floor_counts={str(f): counts[f] for f in range(24, 36)})
    if not directories:
        report["errors"].append("No committed episodes were found")
    if not allow_partial and any(counts[f] != episodes_per_task for f in range(24, 36)):
        report["errors"].append(f"Expected exactly {episodes_per_task} episodes for each of floors 24–35")
    report["success"] = not report["errors"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    references = parser.add_mutually_exclusive_group()
    references.add_argument("--reference", type=Path,
                            help="Single-environment static report (default: lighting_fix_validation/single/report.json)")
    references.add_argument("--reference-raw", type=Path,
                            help="Single-environment schema 9/10 raw root; schemas, rates and fingerprints must match")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path; existing files are not overwritten")
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.episodes_per_task < 1:
        parser.error("--episodes-per-task must be positive")
    if args.output.exists():
        parser.error(f"Output already exists; use a new report path to preserve prior evidence: {args.output}")
    result = audit(args.raw, args.reference, args.episodes_per_task, args.allow_partial, args.reference_raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("episodes", "errors")}, indent=2))
    print(f"Errors: {len(result['errors'])}; full report: {args.output.resolve()}")
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
