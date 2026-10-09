#!/usr/bin/env python3
"""CPU-only provenance, physical-time and decoded-pixel QA for XYZ smoothing videos.

The selected clips must be the first centered repeat of all twelve floors from
the two complete 120-task evaluations, including every success and failure.
Filtering mathematics and aggregate smoothness metrics are audited separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

import numpy as np

import audit_residual_on_noise_video as common

METHODS = ("residual_xyz", "residual_xyz_smoothed")
PAIR = "xyz_vs_smoothed"
EXPECTED = {(method, floor) for method in METHODS for floor in range(24, 36)}
SCOPE = ("First centered repeat for every floor selected from the recorded 120-task "
         "confirmation evaluations; no outcome-based selection.")
read, lines, require, sha = common.read, common.lines, common.require, common.sha


def resolve(value, root):
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve(strict=True)


def write(root, stage, record):
    record.update(created_at=datetime.now(timezone.utc).isoformat(), auditor_sha256=sha(__file__))
    path = root / f"{stage}_qa.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({key: value for key, value in record.items()
                      if key not in ("trajectories", "video_files", "physical_time_pixel_checks")},
                     ensure_ascii=False), flush=True)


def trajectories(root):
    plan = read(root / "video_plan.json")
    selected = plan["selected_trajectories"]
    require(len(selected) == 24, "Expected exactly 24 preselected trajectories")
    require(set(plan["source_runs"]) == set(METHODS), "Wrong policy aliases")
    manifests, episodes, verifications, postprocessing = {}, {}, {}, {}
    for method in METHODS:
        source = resolve(plan["source_runs"][method], root)
        manifest, summary = read(source / "manifest.json"), read(source / "summary.json")
        composition, frozen = read(source / "composition.json"), read(source / "freeze_verification.json")
        post = read(source / "postprocessing.json")
        rows = lines(source / "episodes.jsonl")
        require(summary["state"] == "complete" and summary["episodes"] == len(rows) == 120,
                "Source evaluation incomplete")
        require(summary["updates"] == summary["session_updates"] == 0, "Source evaluation learned")
        require(summary["successes"] == sum(row["success"] for row in rows), "Evaluation success mismatch")
        require(Counter(row["layout"]["floor"] for row in rows) == Counter({floor: 10 for floor in range(24, 36)}),
                "Source evaluation floor coverage differs")
        reference = Path(manifest["checkpoint"]).resolve().parent.parent / "eval"
        verifications[method] = common.verify_reference(manifest, composition, frozen, reference)
        require(post["freeze_verification_passed"] and post["evaluation_updates"] == 0,
                "Postprocessing freeze verification missing")
        require(post["residual_actor_sha256_at_start"] == post["residual_actor_sha256_at_end"] ==
                frozen["residual_actor_sha256"], "Postprocessing actor mismatch")
        require(post["frozen_noise_actor_sha256"] == frozen["noise_actor_sha256"], "Postprocessing noise mismatch")
        require(post["residual_updates_at_start"] == post["residual_updates_at_end"] == 398000,
                "Source actor update count differs")
        require(post["residual_scale"] == [.02] * 3 and post["learned_parameters_updated"] is False and
                post["training_or_checkpoint_modified"] is False, "Postprocessing changed learned policy")
        require(Path(post["residual_checkpoint"]).resolve() == Path(manifest["checkpoint"]).resolve(),
                "Postprocessing checkpoint mismatch")
        require(manifest["learner_config"]["residual_mode"] == "xyz", "Expected XYZ residual")
        if method == METHODS[0]:
            require(post["mode"] == "none" and post["alpha"] == 1, "Left policy is not unfiltered")
        else:
            require(post["mode"] in ("xyz_ema", "pose_ema") and 0 < post["alpha"] < 1,
                    "Right policy is not smoothing")
        index_by_episode = {}
        for row in lines(source / "action_postprocessing.jsonl"):
            identity = row["episode_id"]
            require(identity not in index_by_episode or index_by_episode[identity] == row["episode_index"],
                    "Episode index changed during rollout")
            index_by_episode[identity] = row["episode_index"]
        require(len(index_by_episode) == 120 and set(index_by_episode.values()) == set(range(120)),
                "Missing or repeated evaluation schedule indices")
        episodes[method] = {index_by_episode[row["episode_id"]]: row for row in rows}
        manifests[method], postprocessing[method] = manifest, post
    require(manifests[METHODS[0]]["identities"] == manifests[METHODS[1]]["identities"],
            "Compared policies differ in model, noise or simulator identity")
    require(verifications[METHODS[0]]["actor_sha256"] == verifications[METHODS[1]]["actor_sha256"],
            "Compared policies have different actors")
    require(manifests[METHODS[0]]["config"] == manifests[METHODS[1]]["config"],
            "Compared evaluations differ beyond explicit postprocessing")
    records, initials, sample_counts, source_indexes = {}, {}, Counter(), {}
    max_clock_error = 0.
    for row in selected:
        method, floor = row["method"], int(row["floor"])
        key = (method, floor)
        require(key in EXPECTED and key not in records, "Unexpected or duplicate selected task")
        directory = resolve(row["directory"], root)
        meta = read(directory / "metadata.json")
        manifest = manifests[method]
        episode = episodes[method][floor - 24]
        require(meta["run_id"] == manifest["run_id"], "Trajectory belongs to a different evaluation")
        require(meta["env_id"] == episode["env_id"] == floor - 24, "Not the first centered repeat")
        require(meta["seed"] == manifest["config"]["seed"] + manifest["simulation"]["num_envs"],
                "Trajectory is not from the initial evaluation reset")
        require(meta["reset_episode"] == episode["layout"], "Trajectory/evaluation layout mismatch")
        require(meta["reset_episode"]["floor"] == floor and
                meta["reset_episode"]["offset_x_m"] == meta["reset_episode"]["offset_y_m"] == 0,
                "Selected task is not centered")
        require(meta["schema_version"] == 1 and meta["status"] == "complete", "Incomplete trajectory")
        require(meta["capture_hz"] == 120 and meta["physics_dt"] == 1 / 120, "Wrong capture rate")
        for field, value in meta["info"].items():
            require(episode[field] == value, f"Trajectory/evaluation terminal mismatch: {field}")
        recording_root = directory.parent.parent
        if recording_root not in source_indexes:
            recording = read(recording_root / "trajectory_manifest.json")
            require(recording["capture_hz"] == 120 and
                    recording["additional_physics_steps"] == recording["additional_rgb_captures"] == 0,
                    "Recording changed simulation cadence")
            require(recording["policy_observation_render_cadence"] == "unchanged_chunk_boundary_and_exact_terminal",
                    "Recording changed policy observation cadence")
            for value in recording["extensions"].values():
                require(sha(value["path"]) == value["sha256"], "Recording extension changed")
            source_indexes[recording_root] = lines(recording_root / "index.jsonl")
        run_entries = [item for item in source_indexes[recording_root] if item["run_id"] == meta["run_id"]]
        require(len(run_entries) == 120 and all(item["status"] == "complete" for item in run_entries),
                "Source evaluation is not fully recorded")
        first_per_floor = min((item for item in run_entries if item["floor"] == floor),
                              key=lambda item: item["trajectory_id"])
        require(Path(first_per_floor["directory"]).resolve() == directory,
                "Selected trajectory was not the first scheduled occurrence")
        physics = directory / "physics.npz"
        require(sha(physics) == meta["physics_file"]["sha256"] and
                physics.stat().st_size == meta["physics_file"]["bytes"], "Physics file changed")
        with np.load(physics, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        n = meta["samples"]
        require(n > 1 and set(arrays) == set(meta["arrays"]), "Missing recorded arrays")
        for name, value in arrays.items():
            require(list(value.shape) == meta["arrays"][name]["shape"] and
                    str(value.dtype) == meta["arrays"][name]["dtype"] and
                    value.shape[0] == n and np.isfinite(value).all(), "Invalid physical samples")
        require(np.array_equal(arrays["physics_index"], np.arange(n)), "Missing physical ticks")
        require(n - 1 == meta["last_physics_index"] == meta["info"]["physics_index"], "Missing terminal tick")
        require(np.allclose(arrays["sim_time"], np.arange(n) / 120, rtol=0, atol=1e-12), "Wrong simulation time")
        error = float(np.max(np.abs(np.diff(arrays["world_time"]) - 1 / 120)))
        require(error < 1e-7, "Wrong physics clock intervals")
        max_clock_error = max(max_clock_error, error)
        initial = {field: arrays[field][0].astype(np.float64) for field in
                   ("q_actual", "qd_actual", "state", "button_position_world",
                    "button_orientation_wxyz", "button_velocity_world")}
        initial["button_position_world"] -= np.asarray(meta["env_offset_m"])
        initials[key] = initial
        sample_counts[method] += n
        records[key] = dict(method=method, floor=floor, trajectory=str(directory),
            trajectory_sha256=sha(physics), metadata_sha256=sha(directory / "metadata.json"),
            samples=n, last_tick=n - 1, seed=meta["seed"], episode_index=floor - 24,
            **{field: meta["info"][field] for field in
               ("success", "termination", "pressed_floors", "sim_seconds")})
    require(set(records) == EXPECTED, "Missing preselected tasks")
    differences = {field: max(float(np.max(np.abs(initials[(METHODS[0], floor)][field] -
        initials[(METHODS[1], floor)][field]))) for floor in range(24, 36)) for field in next(iter(initials.values()))}
    require(all(value == 0 for value in differences.values()), f"Different initial physical states: {differences}")
    write(root, "trajectory", dict(status="pass", evidence_origin=SCOPE, episodes=24, methods=list(METHODS),
        video_plan_sha256=sha(root / "video_plan.json"), all_first_center_repeats_retained=True,
        outcome_based_selection=False, zero_parameter_updates=True, policy_verifications=verifications,
        postprocessing={method: {key: postprocessing[method][key] for key in ("mode", "alpha")} for method in METHODS},
        initial_state_max_difference_across_methods=differences, max_world_tick_error_seconds=max_clock_error,
        physics_sample_counts=dict(sample_counts), successes={method: sum(record["success"] for key, record in
        records.items() if key[0] == method) for method in METHODS}, trajectories=[records[key] for key in sorted(records)]))


def renders(root):
    manifest = read(root / "renders/render_manifest.json")
    source_qa = read(root / "trajectory_qa.json")
    require(source_qa["status"] == "pass" and source_qa["video_plan_sha256"] == sha(root / "video_plan.json"),
            "Trajectory audit absent or selection changed")
    sources = {(row["method"], row["floor"]): row for row in source_qa["trajectories"]}
    require(manifest["status"] == "complete" and manifest["expected_episodes"] == len(manifest["episodes"]) == 24,
            "Rendering incomplete")
    require(manifest["physics_steps_during_replay"] == 0 and
            manifest["renderer_initial_world_time"] == manifest["renderer_final_world_time"], "Replay stepped physics")
    require(manifest["fps"] == 30 and manifest["source_physics_hz"] == 120, "Wrong render cadence")
    require(sha(manifest["source_script"]["path"]) == manifest["source_script"]["sha256"], "Renderer source changed")
    seen, videos, maxima = set(), [], {}
    for row in manifest["episodes"]:
        key = (row["method"], row["floor"])
        require(key in EXPECTED and key not in seen, "Unexpected render")
        seen.add(key)
        source = sources[key]
        require(Path(row["source_trajectory"]).resolve() == Path(source["trajectory"]), "Wrong rendered trajectory")
        for field, filename, reference in (("source_trajectory_sha256", "physics.npz", "trajectory_sha256"),
                                           ("source_metadata_sha256", "metadata.json", "metadata_sha256")):
            require(row[field] == source[reference] == sha(Path(source["trajectory"]) / filename), "Render source changed")
        ticks = list(range(0, source["last_tick"] + 1, 4))
        if ticks[-1] != source["last_tick"]:
            ticks.append(source["last_tick"])
        require(row["physics_indices"] == ticks and row["frame_count"] == len(ticks), "Wrong recorded frame samples")
        require(np.allclose(row["frame_sim_seconds"], np.asarray(ticks) / 120, rtol=0, atol=1e-12), "Wrong frame timestamps")
        require(row["physics_steps_during_replay"] == 0 and row["renderer_world_time_start"] == row["renderer_world_time_end"],
                "Episode replay stepped physics")
        for index, (frame, tick) in enumerate(zip(row["frame_records"], ticks, strict=True)):
            require(frame["frame"] == index and frame["physics_index"] == tick and
                    abs(frame["video_time"] - index / 30) < 1e-12 and abs(frame["sim_time"] - tick / 120) < 1e-12,
                    "Wrong frame record")
        require(row["frame_records"][-1]["terminal"] is True, "Terminal state absent")
        for field in ("success", "termination", "pressed_floors", "sim_seconds", "seed"):
            require(row[field] == source[field], "Render changed episode information")
        for field, value in row["restoration_checks"].items():
            require(math.isfinite(value) and 0 <= value <= 2e-6, "Physical state restoration mismatch")
            maxima[field] = max(maxima.get(field, 0), value)
        for view in ("global", "wrist"):
            video = row["videos"][view]
            require(sha(video["path"]) == video["sha256"], "Rendered video changed")
            common.video_info(video["path"], len(ticks), 640, 480)
            videos.append(dict(method=key[0], floor=key[1], view=view, **video))
    require(seen == EXPECTED and len(videos) == 48, "Missing rendered videos")
    write(root, "render", dict(status="pass", episodes=24, videos=48, physics_steps_during_replay=0,
        render_manifest_sha256=sha(root / "renders/render_manifest.json"), restoration_max_errors=maxima,
        all_exact_terminal_frames_retained=True, video_files=videos))


def composition(root):
    manifest_path = root / "renders/render_manifest.json"
    render_qa = read(root / "render_qa.json")
    require(render_qa["status"] == "pass" and render_qa["render_manifest_sha256"] == sha(manifest_path),
            "Render audit absent or render manifest changed")
    rendered = read(manifest_path)
    record = read(root / "videos/composition_manifest.json")
    normalized = read(root / "videos/normalized_input.json")
    require(record["source_manifest_sha256"] == sha(manifest_path) and
            Path(record["source_manifest"]).resolve() == manifest_path, "Wrong composition input")
    require(record["script_sha256"] == sha(Path(__file__).with_name("compose_rl_comparison_videos.py")),
            "Compositor source changed")
    require(record["scope"] == SCOPE, "Composition scope does not identify actual confirmation trajectories")
    require(record["methods"] == normalized["methods"] == list(METHODS), "Wrong comparison columns")
    require(record["fps"] == 30 and record["dimensions"] == [1280, 1152], "Wrong comparison dimensions/rate")
    require([row["floor"] for row in record["cases"]] == [row["floor"] for row in normalized["cases"]]
            == list(range(24, 36)), "Missing or reordered floors")
    render_rows = {(row["method"], row["floor"]): row for row in rendered["episodes"]}
    total, timeline = 0, []
    for case, normalized_case in zip(record["cases"], normalized["cases"], strict=True):
        floor = case["floor"]
        require(case["case_id"] == normalized_case["case_id"] and case["methods"] == list(METHODS), "Wrong case mapping")
        terminal = max(render_rows[(method, floor)]["sim_seconds"] for method in METHODS)
        count = math.ceil(terminal * 30 - 1e-8) + round(record["end_hold_seconds"] * 30) + 1
        require(case["fps"] == 30 and case["common_terminal_seconds"] == terminal and
                case["end_hold_seconds"] == record["end_hold_seconds"] and case["frame_count"] == count and
                abs(case["duration_seconds"] - count / 30) < 1e-12, "Composition changed physical duration")
        for method in METHODS:
            source = render_rows[(method, floor)]
            supplied = normalized_case["methods"][method]
            require(supplied["frame_sim_seconds"] == source["frame_sim_seconds"] and
                    supplied["sim_seconds"] == source["sim_seconds"] and supplied["seed"] == source["seed"],
                    "Composition changed frame timestamps or reset identity")
            for field in ("success", "termination", "pressed_floors", "sim_seconds"):
                require(case["outcomes"][method][field] == source[field], "Composition changed measured outcome")
            for view in ("global", "wrist"):
                supplied_video, rendered_video = case["sources"][method][view], source["videos"][view]
                require(Path(supplied_video["path"]).resolve() == Path(rendered_video["path"]).resolve() and
                        supplied_video["sha256"] == rendered_video["sha256"] == sha(rendered_video["path"]),
                        "Composition substituted camera source")
        require(set(case["results"]) == {"comparison", *METHODS, PAIR}, "Missing case outputs")
        for name, video in case["results"].items():
            common.video_info(video["path"], count, 640 if name in METHODS else 1280, 1152)
        timeline.append(dict(floor=floor, start_seconds=total / 30, duration_seconds=count / 30))
        total += count
    require(set(record["collections"]) == {"comparison", *METHODS, PAIR}, "Missing complete videos")
    for name, video in record["collections"].items():
        require(sha(video["path"]) == video["sha256"], "Complete video changed")
        common.video_info(video["path"], total, 640 if name in METHODS else 1280, 1152)
    common.METHODS, common.MAIN_VIDEO_KEY = METHODS, PAIR
    pixel_checks = common.check_visible_frame_times(record["cases"], render_rows)
    final = record["collections"][PAIR]
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-threads", "1", "-i",
        final["path"], "-map", "0:v:0", "-f", "null", "-"], capture_output=True, text=True)
    require(decoded.returncode == 0 and not decoded.stderr.strip(), f"Full video decode failed: {decoded.stderr}")
    write(root, "composition", dict(status="pass", video=final["path"], sha256=final["sha256"],
        resolution=[1280, 1152], fps=30, frame_count=total, duration_seconds=total / 30,
        same_simulation_time_playback=True, speed_multiplier=1, exact_terminal_states_held=True,
        evidence_origin=SCOPE, physical_time_pixel_checks=pixel_checks, full_decode_exit_code=0, timeline=timeline))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="videos_selected directory")
    parser.add_argument("--stage", choices=("trajectories", "renders", "composition", "all"), default="all")
    args = parser.parse_args()
    root = args.run.resolve(strict=True)
    stages = dict(trajectories=trajectories, renders=renders, composition=composition)
    for name in stages if args.stage == "all" else (args.stage,):
        stages[name](root)


if __name__ == "__main__":
    main()
