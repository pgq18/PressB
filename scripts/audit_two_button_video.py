#!/usr/bin/env python3
"""CPU-only audit of fresh Original VLA versus XYZ EMA clips for buttons 25/31.

Both policies record the original twelve centered tasks. Only the two buttons
requested before recording are rendered. The reference XYZ checkpoint remains
bound to its original completed 400k train and 120-task evaluation artifacts.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

import audit_residual_on_noise_video as common
from analyze_rl_smoothing import audit_action_records

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("base", "residual_xyz_smoothed")
FLOORS = (25, 31)
EXPECTED = {(method, floor) for method in METHODS for floor in FLOORS}
read, lines, require, sha = common.read, common.lines, common.require, common.sha


def resolve(value, root):
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve(strict=True)


def write(root, stage, record):
    record.update(created_at=datetime.now(timezone.utc).isoformat(),
                  auditor_sha256=sha(__file__),
                  audit_dependencies={path.name: sha(path) for path in
                    (Path(common.__file__), ROOT / "scripts/analyze_rl_smoothing.py")})
    path = root / f"{stage}_qa.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({key: value for key, value in record.items()
                     if key not in ("trajectories", "video_files", "physical_time_pixel_checks")},
                    ensure_ascii=False), flush=True)


def verify_sources(manifest):
    for name, digest in manifest["sources"].items():
        require(sha(ROOT / "src/pressb/online_rl" / name) == digest,
                f"Runner source changed: {name}")
    for name, digest in manifest["identities"]["simulation"]["source_sha256"].items():
        require(sha(ROOT / name) == digest, f"Simulation source changed: {name}")
    inference = manifest["identities"]["inference"]["source_identity"]
    require(sha(ROOT / "src/pressb/online_rl/inference.py") == inference["addon_sha256"],
            "Runtime inference adapter differs from audited source")
    for name, digest in inference["source_sha256"].items():
        require(sha(Path(inference["vla_repo"]) / name) == digest,
                f"VLA source changed: {name}")


def checkpoint_actor(path):
    """Read actor tensors on CPU; mmap avoids loading replay arrays into RAM."""
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    digest = hashlib.sha256()
    for name, value in sorted(payload["actor"].items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return payload, digest.hexdigest()


def trajectories(root):
    plan = read(root / "video_plan.json")
    declared = read(root / "plan.json")
    require(all(plan[name] == value for name, value in declared.items()),
            "Final selection changed the predeclared recording plan")
    require(declared["floors"] == list(FLOORS) and declared["methods"] == list(METHODS) and
            declared["seed"] == 20260930 and declared["episodes_per_method"] == declared["num_envs"] == 12 and
            declared["smoothing_mode"] == "xyz_ema" and declared["smoothing_alpha"] == .8,
            "Recording plan changed requested buttons or policy setup")
    for name, digest in declared["source_sha256"].items():
        require(sha(ROOT / name) == digest, f"Recording orchestration source changed: {name}")
    selected = plan["selected_trajectories"]
    require(len(selected) == 4 and {(row["method"], row["floor"]) for row in selected} == EXPECTED,
            "Selection must contain the four requested method/button pairs")
    require(set(plan["source_runs"]) == set(METHODS), "Wrong policy aliases")
    manifests, summaries, episodes, sources = {}, {}, {}, {}
    for method in METHODS:
        source = resolve(plan["source_runs"][method], root)
        manifest, summary = read(source / "manifest.json"), read(source / "summary.json")
        rows = lines(source / "episodes.jsonl")
        require(summary["state"] == "complete" and summary["episodes"] == len(rows) == 12,
                "Fresh evaluation must contain twelve completed tasks")
        require(summary["updates"] == summary["session_updates"] == 0, "Evaluation learned")
        require(summary["successes"] == sum(row["success"] for row in rows), "Success total differs")
        require(len({row["episode_id"] for row in rows}) == 12 and
                Counter(row["layout"]["floor"] for row in rows) == Counter(range(24, 36)),
                "Missing or repeated source task")
        require(manifest["config"]["mode"] == "eval" and manifest["config"]["seed"] == 20260930 and
                manifest["config"]["eval_episodes"] == manifest["simulation"]["num_envs"] == 12,
                "Fresh evaluation schedule differs")
        require(manifest["base_parameters_updated"] is False, "VLA parameters changed")
        require(summary["run_id"] == manifest["run_id"], "Run identity differs")
        verify_sources(manifest)
        for row in rows:
            require(row["layout"]["offset_x_m"] == row["layout"]["offset_y_m"] == 0,
                    "A source task is not centered")
            require(type(row["terminated"]) is bool and type(row["truncated"]) is bool and
                    row["terminated"] != row["truncated"], "Missing terminal/truncation flag")
            require(row["success"] == row["task_success"] == (row["termination"] == "target_pressed"),
                    "Terminal success flags disagree")
        sources[method], manifests[method], summaries[method] = source, manifest, summary
        episodes[method] = {row["layout"]["floor"]: row for row in rows}

    base = manifests["base"]
    require(base["config"]["method"] == "base" and base["learner_config"] is None and
            base["checkpoint"] is None and base["warmstart_actor"] is None,
            "Original VLA loaded an RL learner or checkpoint")
    require("frozen_initial_noise" not in base["identities"] and
            summaries["base"]["rpc_counts"].get("inference_decode", 0) == 0,
            "Original VLA used the learned noise policy")
    require(not (sources["base"] / "postprocessing.json").exists(), "Original VLA was postprocessed")
    for name in ("runner_contract", "inference", "simulation"):
        require(base["identities"][name] == manifests[METHODS[1]]["identities"][name],
                f"Physical or VLA identity differs: {name}")

    source, manifest = sources[METHODS[1]], manifests[METHODS[1]]
    composition, frozen = read(source / "composition.json"), read(source / "freeze_verification.json")
    reference = Path(manifest["checkpoint"]).resolve().parent.parent / "eval"
    require(reference.resolve() == resolve(declared["evaluation_reference"], root) and
            Path(manifest["checkpoint"]).resolve() == resolve(declared["residual_checkpoint"], root),
            "Residual checkpoint or original evaluation differs from recording plan")
    verified = common.verify_reference(manifest, composition, frozen, reference)
    require(verified["residual_mode"] == "xyz" and verified["residual_scale"] == [.02] * 3,
            "Expected the original XYZ-only residual")
    checkpoint, actor = checkpoint_actor(Path(manifest["checkpoint"]))
    require(actor == frozen["residual_actor_sha256"] and checkpoint["updates"] == 398000 and
            checkpoint["extra"]["identities"] == manifest["identities"],
            "Actual residual checkpoint bytes differ from evaluation identity")
    noise_path = Path(composition["frozen_noise"]["checkpoint"])
    require(noise_path.resolve() == resolve(declared["noise_checkpoint"], root),
            "Frozen noise checkpoint differs from recording plan")
    noise, actor = checkpoint_actor(noise_path)
    require(actor == frozen["noise_actor_sha256"] and
            sha(noise_path) == composition["frozen_noise"]["checkpoint_sha256"],
            "Actual frozen noise checkpoint differs")
    post = read(source / "postprocessing.json")
    require(post["mode"] == "xyz_ema" and post["alpha"] == .8 and post["control_hz"] == 30,
            "Wrong smoothing selection")
    require(post["orientation_rule"] == post["gripper_rule"] == "exactly unchanged",
            "Smoothing changes rotation or gripper")
    require(post["freeze_verification_passed"] is True and post["evaluation_updates"] == 0 and
            post["learned_parameters_updated"] is False and post["training_or_checkpoint_modified"] is False,
            "Postprocessing changed learned parameters")
    require(post["residual_actor_sha256_at_start"] == post["residual_actor_sha256_at_end"] ==
            frozen["residual_actor_sha256"] and post["frozen_noise_actor_sha256"] == frozen["noise_actor_sha256"],
            "Postprocessing actor identity differs")
    require(post["residual_updates_at_start"] == post["residual_updates_at_end"] == 398000 and
            post["residual_scale"] == [.02] * 3 and
            Path(post["residual_checkpoint"]).resolve() == Path(manifest["checkpoint"]).resolve(),
            "Postprocessing checkpoint binding differs")
    for name, digest in post["source_sha256"].items():
        require(sha(ROOT / "src/pressb/online_rl" / name) == digest, "Postprocessing source changed")
    action_rows = lines(source / "action_postprocessing.jsonl")
    action_qa = audit_action_records(action_rows, expected_mode="xyz_ema", expected_alpha=.8)
    require(action_qa["episodes"] == action_qa["resets"] == 12 and
            {row["episode_id"] for row in action_rows} == {row["episode_id"] for row in episodes[METHODS[1]].values()} and
            {row["episode_index"] for row in action_rows} == set(range(12)), "Action records miss source episodes")

    records, initials, sample_counts, indexes = {}, {}, Counter(), {}
    max_clock_error = 0.
    for row in selected:
        method, floor = row["method"], int(row["floor"])
        directory = resolve(row["directory"], root)
        meta, episode = read(directory / "metadata.json"), episodes[method][floor]
        require(meta["run_id"] == manifests[method]["run_id"], "Trajectory belongs to another policy run")
        require(meta["env_id"] == episode["env_id"] == floor - 24 and meta["seed"] == 20260942,
                "Trajectory slot or physical reset seed changed")
        require(meta["reset_episode"] == episode["layout"], "Selected layout differs")
        require(meta["status"] == "complete" and meta["capture_hz"] == 120 and meta["physics_dt"] == 1 / 120,
                "Incomplete trajectory or wrong recording frequency")
        for name, value in meta["info"].items():
            require(episode[name] == value, f"Physical terminal information differs: {name}")
        recording_root = directory.parent.parent
        if recording_root not in indexes:
            recording = read(recording_root / "trajectory_manifest.json")
            require(recording["capture_hz"] == 120 and
                    recording["additional_physics_steps"] == recording["additional_rgb_captures"] == 0 and
                    recording["policy_observation_render_cadence"] == "unchanged_chunk_boundary_and_exact_terminal",
                    "Recording changed policy or physics cadence")
            for value in recording["extensions"].values():
                require(sha(value["path"]) == value["sha256"], "Recording extension changed")
            indexes[recording_root] = lines(recording_root / "index.jsonl")
        entries = [item for item in indexes[recording_root] if item["run_id"] == meta["run_id"]]
        require(len(entries) == 12 and all(item["status"] == "complete" for item in entries) and
                Counter(item["floor"] for item in entries) == Counter(range(24, 36)),
                "Source recording schedule incomplete")
        matching = [item for item in entries if item["floor"] == floor]
        require(len(matching) == 1 and Path(matching[0]["directory"]).resolve() == directory,
                "Selected trajectory is not the unique scheduled task")
        physics = directory / "physics.npz"
        require(sha(physics) == meta["physics_file"]["sha256"] and
                physics.stat().st_size == meta["physics_file"]["bytes"], "Physics bytes changed")
        with np.load(physics, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        n = meta["samples"]
        require(n > 1 and set(arrays) == set(meta["arrays"]), "Missing physical samples")
        for name, value in arrays.items():
            require(list(value.shape) == meta["arrays"][name]["shape"] and
                    str(value.dtype) == meta["arrays"][name]["dtype"] and value.shape[0] == n and
                    np.isfinite(value).all(), "Invalid physical samples")
        require(np.array_equal(arrays["physics_index"], np.arange(n)) and
                n - 1 == meta["last_physics_index"] == meta["info"]["physics_index"], "Missing physical ticks")
        require(np.allclose(arrays["sim_time"], np.arange(n) / 120, rtol=0, atol=1e-12) and
                abs(meta["info"]["sim_seconds"] - (n - 1) / 120) < 1e-12, "Physical timing differs")
        if method == METHODS[1]:
            commands = [item for item in action_rows if item["episode_id"] == episode["episode_id"]]
            require(len(commands) == math.ceil((n - 1) / 28), "Command chunks differ from executed physics ticks")
            for command in commands:
                tick = command["chunk_index"] * 28
                require(command["episode_index"] == floor - 24 and command["env_id"] == floor - 24 and
                        np.array_equal(np.asarray(command["observation_state_pose8"]), arrays["state"][tick]),
                        "EMA history is not bound to the measured episode observations")
        error = float(np.max(np.abs(np.diff(arrays["world_time"]) - 1 / 120)))
        require(error < 1e-7, "World time cadence differs")
        max_clock_error = max(max_clock_error, error)
        initials[(method, floor)] = {name: arrays[name][0].astype(np.float64) for name in
            ("q_actual", "qd_actual", "state", "button_position_world", "button_orientation_wxyz", "button_velocity_world")}
        initials[(method, floor)]["button_position_world"] -= np.asarray(meta["env_offset_m"])
        sample_counts[method] += n
        records[(method, floor)] = dict(method=method, floor=floor, trajectory=str(directory),
            trajectory_sha256=sha(physics), metadata_sha256=sha(directory / "metadata.json"),
            samples=n, last_tick=n - 1, seed=meta["seed"],
            **{name: episode[name] for name in ("success", "termination", "pressed_floors", "sim_seconds", "terminated", "truncated")})
    differences = {name: max(float(np.max(np.abs(initials[(METHODS[0], floor)][name] -
        initials[(METHODS[1], floor)][name]))) for floor in FLOORS) for name in next(iter(initials.values()))}
    require(all(value == 0 for value in differences.values()), f"Initial physical states differ: {differences}")
    write(root, "trajectory", dict(status="pass", episodes=4, source_evaluation_episodes_per_method=12,
        methods=list(METHODS), floors=list(FLOORS), video_plan_sha256=sha(root / "video_plan.json"),
        predeclared_plan_sha256=sha(root / "plan.json"),
        selection="Both user-requested buttons, unique centered episode per policy, selected without outcomes",
        original_vla=dict(method="base", learner_loaded=False, rl_checkpoint_loaded=False,
            learned_noise_loaded=False, gaussian_flow_draw="ordinary per-sample torch.randn", parameter_updates=0,
            source_hashes_verified=True, inference_decode_rpc_count=0),
        zero_parameter_updates=True, policy_verification=verified, action_postprocessing_audit=action_qa,
        postprocessing=dict(mode="xyz_ema", alpha=.8, rotation_unchanged=True, gripper_unchanged=True),
        initial_state_max_difference_across_methods=differences, max_world_tick_error_seconds=max_clock_error,
        physics_sample_counts=dict(sample_counts), all_exact_terminal_states_retained=True,
        all_source_successes={method: summaries[method]["successes"] for method in METHODS},
        selected_successes={method: sum(row["success"] for (name, _), row in records.items() if name == method)
                            for method in METHODS},
        trajectories=[records[key] for key in sorted(records)]))


def renders(root):
    manifest = read(root / "renders/render_manifest.json")
    source_qa = read(root / "trajectory_qa.json")
    require(source_qa["status"] == "pass" and source_qa["video_plan_sha256"] == sha(root / "video_plan.json"),
            "Trajectory audit missing or selection changed")
    sources = {(row["method"], row["floor"]): row for row in source_qa["trajectories"]}
    require(manifest["status"] == "complete" and manifest["expected_episodes"] == len(manifest["episodes"]) == 4,
            "Exactly the four requested trajectories must be rendered")
    require(manifest["physics_steps_during_replay"] == 0 and
            manifest["renderer_initial_world_time"] == manifest["renderer_final_world_time"], "Replay stepped physics")
    require(manifest["fps"] == 30 and manifest["source_physics_hz"] == 120, "Wrong render cadence")
    require(sha(manifest["source_script"]["path"]) == manifest["source_script"]["sha256"], "Renderer source changed")
    seen, videos, maxima = set(), [], {}
    for row in manifest["episodes"]:
        key = (row["method"], row["floor"])
        require(key in EXPECTED and key not in seen, "Unexpected rendered trajectory")
        seen.add(key)
        source = sources[key]
        require(Path(row["source_trajectory"]).resolve() == Path(source["trajectory"]), "Wrong rendered source")
        for field, filename, reference in (("source_trajectory_sha256", "physics.npz", "trajectory_sha256"),
                                           ("source_metadata_sha256", "metadata.json", "metadata_sha256")):
            require(row[field] == source[reference] == sha(Path(source["trajectory"]) / filename), "Render source changed")
        ticks = list(range(0, source["last_tick"] + 1, 4))
        if ticks[-1] != source["last_tick"]:
            ticks.append(source["last_tick"])
        require(row["physics_indices"] == ticks and row["frame_count"] == len(ticks) and
                np.allclose(row["frame_sim_seconds"], np.asarray(ticks) / 120, rtol=0, atol=1e-12),
                "Rendered samples or physical timestamps changed")
        require(row["physics_steps_during_replay"] == 0 and row["renderer_world_time_start"] == row["renderer_world_time_end"],
                "Episode replay stepped physics")
        for index, (frame, tick) in enumerate(zip(row["frame_records"], ticks, strict=True)):
            require(frame["frame"] == index and frame["physics_index"] == tick and
                    abs(frame["video_time"] - index / 30) < 1e-12 and abs(frame["sim_time"] - tick / 120) < 1e-12,
                    "Frame timing record differs")
        require(row["frame_records"][-1]["terminal"] is True, "Exact terminal frame missing")
        for name in ("success", "termination", "pressed_floors", "sim_seconds", "seed"):
            require(row[name] == source[name], "Render changed measured outcome")
        for name, value in row["restoration_checks"].items():
            require(math.isfinite(value) and 0 <= value <= 2e-6, "Physical state restoration differs")
            maxima[name] = max(maxima.get(name, 0), value)
        for view in ("global", "wrist"):
            video = row["videos"][view]
            require(sha(video["path"]) == video["sha256"], "Rendered video changed")
            common.video_info(video["path"], len(ticks), 640, 480)
            videos.append(dict(method=key[0], floor=key[1], view=view, **video))
    require(seen == EXPECTED and len(videos) == 8, "Missing rendered camera video")
    write(root, "render", dict(status="pass", episodes=4, videos=8, physics_steps_during_replay=0,
        render_manifest_sha256=sha(root / "renders/render_manifest.json"), restoration_max_errors=maxima,
        all_exact_terminal_frames_retained=True, video_files=videos))


def composition(root):
    from PIL import Image

    manifest_path = root / "renders/render_manifest.json"
    render_qa = read(root / "render_qa.json")
    require(render_qa["status"] == "pass" and render_qa["render_manifest_sha256"] == sha(manifest_path),
            "Render audit missing or source manifest changed")
    rendered = read(manifest_path)
    record = read(root / "videos/composition_manifest.json")
    require(record["status"] == "complete", "Composition incomplete")
    for field, path in (("source_manifest", manifest_path),
                         ("source_script", ROOT / "scripts/compose_two_button_comparison.py"),
                         ("shared_reader_script", ROOT / "scripts/compose_rl_comparison_videos.py")):
        source = record[field]
        require(Path(source["path"]).resolve() == path.resolve() and source["sha256"] == sha(path) and
                source["bytes"] == path.stat().st_size, f"Composition dependency changed: {field}")
    require(record["dimensions"] == [1920, 1080] and record["fps"] == 30 and record["playback_speed"] == 1,
            "Composition geometry or playback speed changed")
    require(record["floors_top_to_bottom"] == list(FLOORS) and
            record["methods_left_to_right"] == list(METHODS), "Comparison rows or columns changed")
    sources = {(row["method"], row["floor"]): row for row in rendered["episodes"]}
    terminal = max(row["sim_seconds"] for row in sources.values())
    count = math.ceil((terminal + 1) * 30 - 1e-8)
    require(record["common_terminal_seconds"] == terminal and record["end_hold_seconds"] == 1 and
            record["frame_count"] == count and record["duration_seconds"] == count / 30,
            "Physical duration was compressed or terminal hold changed")
    require(record["output_frame_seconds"] == [index / 30 for index in range(count)],
            "Output frames use a changed time axis")
    require(abs(record["requested_duration_seconds"] - (terminal + 1)) < 1e-12 and
            abs(record["duration_rounding_seconds"] - (count / 30 - terminal - 1)) < 1e-12 and
            -1e-12 <= record["duration_rounding_seconds"] < 1 / 30, "Unexpected duration rounding")
    video = record["video"]
    require(sha(video["path"]) == video["sha256"] and Path(video["path"]).stat().st_size == video["bytes"],
            "Final video changed")
    common.video_info(video["path"], count, 1920, 1080)
    geometry = {}
    require(len(record["geometry"]) == 2, "Missing simultaneous task row")
    for row_index, row in enumerate(record["geometry"]):
        require(row["row"] == row_index and row["floor"] == FLOORS[row_index] and len(row["cells"]) == 2,
                "Geometry rows reordered")
        for column, cell in enumerate(row["cells"]):
            require(cell["column"] == column and cell["method"] == METHODS[column], "Geometry columns reordered")
            for view_index, view in enumerate(("global", "wrist")):
                expected = [column * 960 + view_index * 480, 188 + row_index * 468, 480, 360]
                require(cell[f"{view}_xywh"] == expected, "Camera display crop or placement changed")
                geometry[(METHODS[column], FLOORS[row_index], view)] = expected
    seen, checks, composed_cache, endpoints = set(), [], {}, []
    for row in record["episodes"]:
        key = (row["method"], row["floor"])
        require(key in EXPECTED and key not in seen, "Unexpected composed trajectory")
        seen.add(key)
        source = sources[key]
        require(row["row"] == FLOORS.index(key[1]) and row["column"] == METHODS.index(key[0]),
                "Trajectory shown in wrong cell")
        require(row["source_frame_seconds"] == source["frame_sim_seconds"] and
                row["source_physics_indices"] == source["physics_indices"], "Source timing changed")
        expected_mapping = [max(0, bisect_right(source["frame_sim_seconds"], index / 30 + 1e-9) - 1)
                            for index in range(count)]
        require(row["source_frame_for_each_output_frame"] == expected_mapping,
                "Composition skips, anticipates, or retimes physical frames")
        terminal_frame = expected_mapping.index(source["frame_count"] - 1)
        require(row["first_terminal_output_frame"] == terminal_frame and
                row["first_terminal_output_seconds"] == terminal_frame / 30 and
                abs(row["terminal_display_delay_seconds"] - (terminal_frame / 30 - source["sim_seconds"])) < 1e-12 and
                -1e-12 <= row["terminal_display_delay_seconds"] < 1 / 30 and
                row["held_terminal_frame_count"] == count - terminal_frame and
                row["held_terminal_duration_seconds"] == (count - terminal_frame) / 30,
                "Exact terminal state is delayed or not held")
        require(all(index == source["frame_count"] - 1 for index in expected_mapping[terminal_frame:]),
                "Motion continues after physical termination")
        for name in ("success", "termination", "pressed_floors", "sim_seconds"):
            require(row["outcome"][name] == source[name], "Displayed measured outcome changed")
        for name in ("source_trajectory", "source_trajectory_sha256", "source_metadata_sha256", "seed"):
            require(row["source_identity"][name] == source[name], f"Composition source identity changed: {name}")
        for view in ("global", "wrist"):
            declared, original = row["source_videos"][view], source["videos"][view]
            require(Path(declared["path"]).resolve() == Path(original["path"]).resolve() and
                    declared["sha256"] == original["sha256"] == sha(original["path"]),
                    "Composition substituted a camera video")
        for label, output_index in (("initial", 0), ("motion", round(source["sim_seconds"] * 30 / 2)),
                                    ("exact_terminal", terminal_frame), ("final_hold", count - 1)):
            if output_index not in composed_cache:
                composed_cache[output_index] = common.decoded_frame(video["path"], output_index, 1920, 1080)
            composed = composed_cache[output_index]
            source_index = expected_mapping[output_index]
            for view in ("global", "wrist"):
                raw = common.decoded_frame(source["videos"][view]["path"], source_index, 640, 480)
                measured = np.asarray(Image.fromarray(raw).resize((480, 360), Image.Resampling.LANCZOS))
                left, top, width, height = geometry[(*key, view)]
                displayed = composed[top:top + height, left:left + width]
                # Exclude the view-name badge and edge ringing from H.264.
                difference = np.abs(displayed[50:350, 8:472].astype(np.int16) -
                                    measured[50:350, 8:472].astype(np.int16))
                error = float(difference.mean())
                require(error < 5, f"Displayed pixels differ from physical time: {key} {view} {label}: {error}")
                checks.append(dict(method=key[0], floor=key[1], view=view, check=label,
                    output_frame=output_index, output_seconds=output_index / 30,
                    source_frame=source_index, source_physical_seconds=source["frame_sim_seconds"][source_index],
                    mean_absolute_rgb_error=error))
        endpoints.append(dict(method=key[0], floor=key[1], sim_seconds=source["sim_seconds"],
                              terminal_output_frame=terminal_frame, held_terminal_frames=count - terminal_frame))
    require(seen == EXPECTED and len(checks) == 32, "Missing simultaneous trajectories or pixel checks")
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-threads", "1", "-i",
        video["path"], "-map", "0:v:0", "-f", "null", "-"], capture_output=True, text=True)
    require(decoded.returncode == 0 and not decoded.stderr.strip(), f"Final video decode failed: {decoded.stderr}")
    write(root, "composition", dict(status="pass", video=video["path"], sha256=video["sha256"],
        resolution=[1920, 1080], fps=30, frame_count=count, duration_seconds=count / 30,
        floors_top_to_bottom=list(FLOORS), methods_left_to_right=list(METHODS),
        same_simulation_time_playback=True, speed_multiplier=1, exact_terminal_states_held=True,
        all_four_trajectories_simultaneous=True, source_frame_mappings_verified=True,
        physical_time_pixel_checks=checks, full_decode_exit_code=0, terminal_frames=endpoints))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=("trajectories", "renders", "composition", "all"), default="all")
    args = parser.parse_args()
    root = args.run.resolve(strict=True)
    for stage in ("trajectories", "renders", "composition") if args.stage == "all" else (args.stage,):
        globals()[stage](root)


if __name__ == "__main__":
    main()
