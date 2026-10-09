#!/usr/bin/env python3
"""Audit measured 400k residual-on-noise videos without starting a simulator.

Checks recording provenance, zero-step offline rendering, and real-time video
composition. All 12 centered tasks from both methods must be retained, including
failures. These new illustration episodes are separate from the 120-task eval.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import math
import json
from pathlib import Path
import subprocess

import numpy as np


RUN = None
METHODS = ("base", "residual_on_noise")
MAIN_VIDEO_KEY = "before_vs_residual_on_noise"
REFERENCE = None
DEFAULT_REFERENCE = "online_rl_residual_on_noise_400k_20261006"
EXPECTED = {(method, floor) for method in METHODS for floor in range(24, 36)}


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require(test, message):
    if not test:
        raise ValueError(message)


def configure_plan():
    """Use the declared pair, preserving the original before/after default."""
    global METHODS, EXPECTED, MAIN_VIDEO_KEY
    plan = read(RUN / "plan.json") if (RUN / "plan.json").exists() else {}
    METHODS = tuple(plan.get("methods", ("base", "residual_on_noise")))
    pairs = {("base", "residual_on_noise"): "before_vs_residual_on_noise",
             ("residual_fullscale", "residual_on_noise"): "fullscale_vs_halfscale",
             ("base", "residual_xyz"): "before_vs_residual_xyz",
             ("residual_fullscale", "residual_xyz"): "pose9_vs_xyz"}
    require(METHODS in pairs, "Unsupported or reordered comparison methods")
    MAIN_VIDEO_KEY = plan.get("main_video_key", pairs[METHODS])
    require(MAIN_VIDEO_KEY == pairs[METHODS], "Comparison video key differs from declared methods")
    EXPECTED = {(method, floor) for method in METHODS for floor in range(24, 36)}


def canonical_noise(noise):
    """Old checkpoints omit the unchanged default pose9 action mode."""
    return {**noise, "config": {"residual_mode": "pose9", **noise["config"]}}


def resolve_reference(run, explicit=None, method="residual_on_noise"):
    """Select an eval directory: explicit CLI path, run plan, then legacy run."""
    plan = read(run / "plan.json") if (run / "plan.json").exists() else {}
    references = plan.get("evaluation_references", {})
    if explicit is not None:
        selected = Path(explicit)
    elif method in references:
        selected = Path(references[method])
        if not selected.is_absolute():
            selected = run / selected
    elif method in ("residual_on_noise", "residual_xyz") and plan.get("evaluation_reference"):
        selected = Path(plan["evaluation_reference"])
        if not selected.is_absolute():
            selected = run / selected
    else:
        require(method == "residual_on_noise", f"No evaluation reference declared for {method}")
        selected = run.parent / DEFAULT_REFERENCE / "eval"
    selected = selected.resolve(strict=True)
    require(selected.is_dir(), "Evaluation reference must be an eval directory")
    for filename in ("manifest.json", "summary.json", "composition.json", "freeze_verification.json"):
        require((selected / filename).is_file(), f"Evaluation reference missing {filename}")
    return selected


def verify_reference(policy_manifest, composition, frozen, reference_path):
    """Derive actor identities from the completed reference eval, without literals."""
    reference = read(reference_path / "manifest.json")
    reference_summary = read(reference_path / "summary.json")
    reference_composition = read(reference_path / "composition.json")
    reference_frozen = read(reference_path / "freeze_verification.json")
    require(reference_summary["state"] == "complete" and reference_summary["episodes"] == 120,
            "Reference 120-episode evaluation incomplete")
    require(reference_summary["updates"] == reference_summary["session_updates"] == 0,
            "Reference evaluation updated parameters")
    require(reference["config"]["mode"] == reference_composition["mode"] == composition["mode"] == "eval",
            "Composition or reference not evaluated frozen")
    require(reference["identities"] == policy_manifest["identities"],
            "Recorded policy contract differs from 120-episode evaluation")
    require(frozen == reference_frozen, "Recorded actors differ from 400k final evaluation")
    require(frozen["noise_checkpoint_unchanged"] is True, "Noise checkpoint changed")
    for field in ("base_parameters_updated", "noise_parameters_updated"):
        require(frozen[field] is False and composition[field] is False and
                reference_composition[field] is False, f"Frozen parameters updated: {field}")
    require(frozen["residual_updates"] == composition["residual_updates_at_start"] ==
            reference_composition["residual_updates_at_start"] == 398000,
            "Residual learner updates differ from final checkpoint")
    for actor, field, expected in (
        ("Residual", composition["residual_actor_sha256_at_start"],
         reference_composition["residual_actor_sha256_at_start"]),
        ("Noise", composition["frozen_noise"]["actor_sha256"],
         reference_composition["frozen_noise"]["actor_sha256"]),
    ):
        require(isinstance(expected, str) and len(expected) == 64 and
                all(character in "0123456789abcdef" for character in expected),
                f"Invalid reference {actor.lower()} actor hash")
        require(frozen[f"{actor.lower()}_actor_sha256"] == field == expected,
                f"{actor} actor differs from final evaluation")
    require(canonical_noise(composition["frozen_noise"]) == canonical_noise(reference_composition["frozen_noise"]),
            "Frozen noise sampling contract differs from final evaluation")
    require(composition.get("frozen_noise_objective_transfer") ==
            reference_composition.get("frozen_noise_objective_transfer"),
            "Frozen noise gamma transfer differs from final evaluation")
    require(composition["frozen_noise"]["deterministic"] is False, "Noise sampling changed")
    require(composition["replay_size_at_start"] == reference_composition["replay_size_at_start"] == 0,
            "Evaluation loaded replay")
    require(Path(policy_manifest["checkpoint"]).resolve() == Path(reference["checkpoint"]).resolve(),
            "Different residual checkpoint")
    checkpoint = Path(reference["checkpoint"]).resolve()
    training = (reference_path.parent / "train").resolve()
    require(checkpoint == training / "last.pt" and checkpoint.is_file(),
            "Reference is not the final training checkpoint")
    train_manifest = read(training / "manifest.json")
    train_summary = read(training / "summary.json")
    train_frozen = read(training / "freeze_verification.json")
    train_config = read(reference_path.parent / "train_config.json")
    require(train_summary["state"] == "complete" and train_summary["training_transitions"] == 400000 and
            train_summary["updates"] == 398000, "Reference training budget differs from 400k")
    require(train_frozen == reference_frozen, "Eval actors differ from the final training actors")
    require(train_manifest["identities"] == reference["identities"], "Training/eval identities differ")
    scale = policy_manifest["learner_config"]["residual_scale"]
    mode = policy_manifest["learner_config"].get("residual_mode", "pose9")
    require(mode in ("pose9", "xyz"), "Unsupported residual action mode")
    per_step_dim = 3 if mode == "xyz" else 9
    require(len(scale) == per_step_dim and all(type(value) in (int, float) and math.isfinite(value) and value > 0
                                   for value in scale), "Invalid actual residual scale")
    for label, config, actual in (("recording", policy_manifest["config"], policy_manifest["learner_config"]),
                                  ("reference", reference["config"], reference["learner_config"]),
                                  ("training", train_manifest["config"], train_manifest["learner_config"])):
        require(config["learner"]["residual_scale"] == actual["residual_scale"] == scale,
                f"Configured/actual residual scale mismatch: {label}")
        require(config["learner"].get("residual_mode", "pose9") == actual.get("residual_mode", "pose9") == mode,
                f"Configured/actual residual mode mismatch: {label}")
        require(config["single_gamma"] == reference["identities"]["simulation"]["single_gamma"],
                f"Configured gamma differs from reference: {label}")
    require(train_config["learner"]["residual_scale"] == scale, "Training input scale differs")
    require(train_config["learner"].get("residual_mode", "pose9") == mode, "Training input mode differs")
    for label, declared in (("recording", composition), ("reference", reference_composition)):
        require(declared.get("residual_mode", "pose9") == mode, f"Composition residual mode mismatch: {label}")
        if mode == "xyz":
            require(all(field in declared for field in
                        ("residual_action_dim", "replay_action_dim", "residual_scale")),
                    f"XYZ action dimensions or scale not declared: {label}")
        for field in ("residual_action_dim", "replay_action_dim"):
            require(declared.get(field, 63) == per_step_dim * 7, f"Wrong {field}: {label}")
        require(declared.get("residual_scale", scale) == scale, f"Composition scale mismatch: {label}")
    return dict(evaluation_reference=str(reference_path), training_directory=str(training),
                checkpoint=str(checkpoint), residual_scale=scale, residual_mode=mode,
                residual_action_dim=per_step_dim * 7,
                single_gamma=reference["config"]["single_gamma"],
                actor_sha256={actor: frozen[f"{actor}_actor_sha256"] for actor in ("noise", "residual")})


def verify_policy_pair(manifests, verified):
    for component in ("runner_contract", "inference", "simulation"):
        require(manifests[METHODS[0]]["identities"][component] ==
                manifests[METHODS[1]]["identities"][component],
                f"Policy model/simulator identity differs: {component}")
    if METHODS[0] == "residual_fullscale":
        left, right = (verified[method] for method in METHODS)
        require(manifests[METHODS[0]]["identities"] == manifests[METHODS[1]]["identities"],
                "Compared policies differ in frozen model, noise or physical identity")
        require(left["actor_sha256"]["noise"] == right["actor_sha256"]["noise"] and
                left["actor_sha256"]["residual"] != right["actor_sha256"]["residual"],
                "Comparison requires the same noise and distinct residual actors")
        left_config, right_config = (read(Path(result["training_directory"]).parent / "train_config.json")
                                    for result in (left, right))
        for config in (left_config, right_config):
            config["learner"].pop("residual_mode", None)
            config["learner"].pop("residual_scale")
        require(left_config == right_config, "Training configurations differ beyond residual mode/scale")
        if METHODS[1] == "residual_xyz":
            require(left["residual_mode"] == "pose9" and right["residual_mode"] == "xyz",
                    "Expected pose9 versus XYZ residual comparison")
            require(left["residual_scale"] == [.03] * 3 + [.1] * 6 and
                    right["residual_scale"] == [.02] * 3, "Wrong pose9/XYZ residual scales")
            require(left["single_gamma"] == right["single_gamma"] == .995, "Wrong pose9/XYZ discount")
        else:
            require(left["residual_mode"] == right["residual_mode"] == "pose9",
                    "Full/half-scale comparison changed residual action mode")
            require(all(half == full * .5 for full, half in
                        zip(left["residual_scale"], right["residual_scale"], strict=True)),
                    "Half-scale policy is not exactly 50 percent of the full-scale policy")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(name, value):
    value["created_at"] = datetime.now(timezone.utc).isoformat()
    value["auditor_sha256"] = sha(__file__)
    target = RUN / name
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(target)


def trajectories():
    configure_plan()
    status = read(RUN / "status.json")
    require(status["state"] != "failed", f"Pipeline failed: {status}")
    require(read(RUN / "recording_status.json")["state"] == "complete", "Recording incomplete")
    run_map = read(RUN / "run_map.json")
    require(set(run_map.values()) == set(METHODS), "Unexpected policy runs")
    trajectory_root = RUN / "simulation/trajectories"
    manifest = read(trajectory_root / "trajectory_manifest.json")
    require(manifest["capture_hz"] == 120, "Not a 120 Hz recording")
    require(manifest["additional_physics_steps"] == manifest["additional_rgb_captures"] == 0,
            "Recording changed simulator cadence")
    require(manifest["policy_observation_render_cadence"] == "unchanged_chunk_boundary_and_exact_terminal",
            "Policy render cadence changed")
    for source in manifest["extensions"].values():
        require(sha(source["path"]) == source["sha256"], "Recording extension source changed")
    index = lines(trajectory_root / "index.jsonl")
    require(len(index) == 24, "Recording must contain exactly 24 episodes")
    require(all(row["status"] == "complete" for row in index), "Incomplete trajectory")
    manifests, summaries, episode_maps = {}, {}, {}
    for method in METHODS:
        path = RUN / f"{method}_eval"
        manifests[method] = read(path / "manifest.json")
        summaries[method] = read(path / "summary.json")
        episode_rows = lines(path / "episodes.jsonl")
        require(len(episode_rows) == 12, f"Incorrect episode count: {method}")
        require(len({row["episode_id"] for row in episode_rows}) == 12, "Repeated episode ID")
        episode_maps[method] = {row["layout"]["floor"]: row for row in episode_rows}
        require(set(episode_maps[method]) == set(range(24, 36)), "Missing floor")
        summary, policy_manifest = summaries[method], manifests[method]
        require(summary["state"] == "complete" and summary["episodes"] == 12, "Evaluation incomplete")
        require(summary["updates"] == summary["session_updates"] == 0, "Evaluation updated parameters")
        require(policy_manifest["config"]["seed"] == 20260930, "Unexpected task seed")
        require(policy_manifest["base_parameters_updated"] is False, "Base model not frozen")
        require(policy_manifest["config"]["mode"] == "eval", "Recording ran in training mode")
        require(sum(row["success"] for row in episode_rows) == summary["successes"], "Success mismatch")
    verified = {}
    for method in METHODS:
        if method == "base":
            continue
        reference_path = resolve_reference(RUN, REFERENCE if method == METHODS[1] else None, method)
        policy_directory = RUN / f"{method}_eval"
        verified[method] = verify_reference(manifests[method], read(policy_directory / "composition.json"),
                                           read(policy_directory / "freeze_verification.json"), reference_path)
    verify_policy_pair(manifests, verified)
    plan = read(RUN / "plan.json")
    for method, result in verified.items():
        if "residual_modes" in plan:
            require(plan["residual_modes"][method] == result["residual_mode"], "Mode differs from recording plan")
        if "residual_scales" in plan:
            require(plan["residual_scales"][method] == result["residual_scale"], "Scale differs from recording plan")

    records, initials, sample_counts = {}, {}, Counter()
    max_clock_delta_error = 0.0
    for entry in index:
        directory = Path(entry["directory"])
        meta = read(directory / "metadata.json")
        method = run_map[meta["run_id"]]
        floor = meta["reset_episode"]["floor"]
        key = (method, floor)
        require(key not in records, "Duplicate method/floor")
        episode = episode_maps[method][floor]
        require(meta["status"] == "complete" and entry["status"] == "complete", "Incomplete trajectory")
        require(meta["reset_episode"] == episode["layout"], "Trajectory/evaluation task mismatch")
        require(meta["reset_episode"]["offset_x_m"] == meta["reset_episode"]["offset_y_m"] == 0, "Noncentral layout")
        require(meta["env_id"] == episode["env_id"] == floor - 24, "Condition/environment mismatch")
        require(meta["capture_hz"] == 120 and meta["physics_dt"] == 1 / 120, "Wrong physics rate")
        require(meta["seed"] == 20260942, "Unexpected physical reset seed")
        for field, value in meta["info"].items():
            require(episode[field] == value, f"Trajectory terminal info mismatch: {field}")
        physics = directory / "physics.npz"
        require(sha(physics) == meta["physics_file"]["sha256"], "Trajectory hash mismatch")
        require(physics.stat().st_size == meta["physics_file"]["bytes"], "Trajectory size mismatch")
        with np.load(physics, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        n = meta["samples"]
        require(set(arrays) == set(meta["arrays"]), "Missing array")
        for field, values in arrays.items():
            require(list(values.shape) == meta["arrays"][field]["shape"], "Array shape mismatch")
            require(str(values.dtype) == meta["arrays"][field]["dtype"], "Array dtype mismatch")
            require(values.shape[0] == n and np.isfinite(values).all(), "Missing/nonfinite samples")
        require(np.array_equal(arrays["physics_index"], np.arange(n)), "Missing/repeated physics ticks")
        require(n - 1 == meta["last_physics_index"] == meta["info"]["physics_index"], "Terminal tick mismatch")
        require(np.allclose(arrays["sim_time"], np.arange(n) / 120, rtol=0, atol=1e-12), "Sim time mismatch")
        error = float(np.max(np.abs(np.diff(arrays["world_time"]) - 1 / 120)))
        max_clock_delta_error = max(max_clock_delta_error, error)
        require(error < 1e-7, "World clock increment differs from 120 Hz")
        initial = {field: arrays[field][0].astype(np.float64) for field in
            ("q_actual", "qd_actual", "state", "button_position_world", "button_orientation_wxyz", "button_velocity_world")}
        initial["button_position_world"] -= np.asarray(meta["env_offset_m"])
        initials[key] = initial
        sample_counts[method] += n
        records[key] = dict(method=method, floor=floor, trajectory=str(directory),
            trajectory_sha256=sha(physics), metadata_sha256=sha(directory / "metadata.json"),
            samples=n, last_tick=n - 1, seed=meta["seed"], success=meta["info"]["success"],
            termination=meta["info"]["termination"], pressed_floors=meta["info"]["pressed_floors"],
            sim_seconds=meta["info"]["sim_seconds"])
    require(set(records) == EXPECTED, "Missing or extra selected tasks")
    differences = {}
    for field in next(iter(initials.values())):
        differences[field] = max(float(np.max(np.abs(initials[(METHODS[0], floor)][field] -
            initials[(METHODS[1], floor)][field]))) for floor in range(24, 36))
        require(differences[field] == 0, f"Initial state differs: {field}")
    for method in METHODS:
        require(sample_counts[method] - 12 == summaries[method]["physical_ticks"], "Recorded tick total mismatch")

    qa = dict(status="pass", episodes=24, methods=list(METHODS), floors=list(range(24, 36)),
        all_predeclared_tasks_retained=True, all_successes_and_failures_retained=True,
        initial_state_max_difference_across_methods=differences,
        button_position_comparison="world positions minus each trajectory env_offset_m",
        all_matching_task_layout_and_reset_seed=True, runner_seed=20260930, physical_reset_seed=20260942,
        all_trajectory_hashes_match=True, all_samples_finite_and_contiguous_120hz=True,
        all_exact_terminal_states_retained=True, max_world_tick_error_seconds=max_clock_delta_error,
        physics_sample_counts=dict(sample_counts), recording_extra_physics_steps=0, recording_extra_rgb_captures=0,
        identical_model_and_simulation_identities=True,
        residual_full_identity_matches_final_eval=True, zero_parameter_updates=True,
        evaluation_reference=verified[METHODS[1]]["evaluation_reference"],
        expected_actor_sha256=verified[METHODS[1]]["actor_sha256"],
        policy_verifications=verified,
        evaluation_references={method: result["evaluation_reference"] for method, result in verified.items()},
        full_vs_half_scale_verified=MAIN_VIDEO_KEY == "fullscale_vs_halfscale",
        pose9_vs_xyz_verified=MAIN_VIDEO_KEY == "pose9_vs_xyz",
        frozen_actor_and_checkpoint_verification_pass=True, frozen_noise_sampling_contract_matches_final_eval=True,
        successes={method: summaries[method]["successes"] for method in METHODS},
        trajectories=[records[key] for key in sorted(records)],
        scope="Fresh central-layout illustration rollouts; separate from the original 120-episode evaluation.")
    write("trajectory_qa.json", qa)
    with (RUN / "results.csv").open("w", newline="") as stream:
        columns = ["floor"] + [f"{method}_{field}" for method in METHODS
            for field in ("success", "termination", "pressed_floors", "sim_seconds", "trajectory")]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for floor in range(24, 36):
            writer.writerow(dict(floor=floor, **{f"{method}_{field}": records[(method, floor)][field]
                for method in METHODS for field in
                ("success", "termination", "pressed_floors", "sim_seconds", "trajectory")}))
    print(json.dumps({key: value for key, value in qa.items() if key != "trajectories"}, indent=2))


def renders():
    configure_plan()
    require(read(RUN / "status.json")["state"] != "failed", "Pipeline failed")
    manifest = read(RUN / "renders/render_manifest.json")
    trajectory_qa = read(RUN / "trajectory_qa.json")
    require(trajectory_qa["status"] == "pass", "Trajectory audit failed")
    require(trajectory_qa["methods"] == list(METHODS), "Trajectory audit methods differ from plan")
    require(manifest["status"] == "complete", "Rendering incomplete")
    require(len(manifest["episodes"]) == manifest["expected_episodes"] == 24, "Missing rendered episodes")
    require(manifest["physics_steps_during_replay"] == 0, "Replay integrated physics")
    require(manifest["renderer_initial_world_time"] == manifest["renderer_final_world_time"], "Replay advanced world clock")
    source_records = {(row["method"], row["floor"]): row for row in trajectory_qa["trajectories"]}
    seen, videos, maxima = set(), [], {}
    require(manifest["fps"] == 30 and manifest["source_physics_hz"] == 120, "Render timing changed")
    require(sha(manifest["source_script"]["path"]) == manifest["source_script"]["sha256"],
            "Renderer source changed")
    for episode in manifest["episodes"]:
        key = (episode["method"], episode["floor"])
        require(key in EXPECTED and key not in seen, "Unexpected/duplicate rendered episode")
        seen.add(key)
        source = source_records[key]
        require(episode["source_trajectory"] == source["trajectory"], "Wrong source trajectory")
        for field, filename, expected in [("source_trajectory_sha256", "physics.npz", source["trajectory_sha256"]),
                                          ("source_metadata_sha256", "metadata.json", source["metadata_sha256"])]:
            require(episode[field] == expected == sha(Path(source["trajectory"]) / filename), "Render source hash mismatch")
        require(episode["physics_steps_during_replay"] == 0, "Episode replay integrated physics")
        require(episode["renderer_world_time_start"] == episode["renderer_world_time_end"], "Episode replay advanced time")
        expected_ticks = list(range(0, source["last_tick"] + 1, 4))
        if expected_ticks[-1] != source["last_tick"]:
            expected_ticks.append(source["last_tick"])
        require(episode["physics_indices"] == expected_ticks, "Video sampling dropped/changed measured ticks")
        require(episode["frame_count"] == len(expected_ticks), "Frame count does not match samples")
        require(np.allclose(episode["frame_sim_seconds"], np.array(expected_ticks) / 120,
                            rtol=0, atol=1e-12), "Frame timestamps do not match measured ticks")
        require(len(episode["frame_records"]) == len(expected_ticks), "Missing frame records")
        for frame, tick in zip(episode["frame_records"], expected_ticks, strict=True):
            require(frame["physics_index"] == tick and abs(frame["sim_time"] - tick / 120) < 1e-12,
                    "Rendered frame time differs from physical state")
            require(abs(frame["video_time"] - frame["frame"] / 30) < 1e-12,
                    "Rendered frames are not constant 30 fps")
        require(episode["frame_records"][-1]["terminal"] is True, "Terminal frame missing")
        for field in ("success", "termination", "pressed_floors", "sim_seconds"):
            require(episode[field] == source[field], "Outcome changed during rendering")
        for field, value in episode["restoration_checks"].items():
            maxima[field] = max(maxima.get(field, 0), value)
            require(value < 1e-5, "Restored measured state differs")
        for view in ("global", "wrist"):
            video = episode["videos"][view]
            path = Path(video["path"])
            require(sha(path) == video["sha256"], "Video hash differs")
            info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration", "-of", "json", str(path)]))["streams"][0]
            require(info["width"] == 640 and info["height"] == 480 and info["avg_frame_rate"] == "30/1", "Wrong video dimensions/fps")
            require(int(info["nb_frames"]) == episode["frame_count"], "Encoded video frame count mismatch")
            require(abs(float(info["duration"]) - episode["frame_count"] / 30) < 1e-5, "Video duration mismatch")
            videos.append(dict(method=key[0], floor=key[1], view=view, path=str(path), sha256=video["sha256"],
                               frame_count=int(info["nb_frames"])))
    require(seen == EXPECTED, "Not all tasks rendered")
    qa = dict(status="pass", episodes=24, videos=48, all_source_hashes_match=True,
        all_selected_tasks_and_outcomes_preserved=True, physics_steps_during_replay=0,
        renderer_world_time_start=manifest["renderer_initial_world_time"],
        renderer_world_time_end=manifest["renderer_final_world_time"], clock_unchanged=True,
        restoration_max_errors=maxima, all_videos_640x480_30fps=True, frame_counts_match=True,
        all_exact_terminal_frames_retained=True, video_files=videos,
        initialization_note=manifest["initialization"],
        presentation_note="New RGB renders of measured rollout states; not original policy observation RGB.")
    write("render_qa.json", qa)
    print(json.dumps({key: value for key, value in qa.items() if key != "video_files"}, indent=2))


def video_info(path, frame_count, width, height):
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration", "-of", "json", str(path)
    ]))["streams"][0]
    require(info["width"] == width and info["height"] == height, "Wrong composition dimensions")
    require(info["avg_frame_rate"] == "30/1", "Composition playback rate changed")
    require(int(info["nb_frames"]) == frame_count, "Composition dropped or added frames")
    require(abs(float(info["duration"]) - frame_count / 30) < 1e-5, "Composition duration changed")
    return info


def decoded_frame(path, index, width, height):
    result = subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-i", str(path),
        "-vf", f"select=eq(n\\,{index})", "-frames:v", "1", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-threads", "1", "pipe:1"
    ], check=True, capture_output=True)
    require(len(result.stdout) == width * height * 3, "Could not decode requested comparison frame")
    return np.frombuffer(result.stdout, np.uint8).reshape(height, width, 3)


def check_visible_frame_times(cases, render_rows):
    """Check decoded pixels for previously questioned floors, ignoring overlays.

The composite is H.264 encoded again, so exact byte equality is inappropriate.
The pixel check verifies a mid-motion and exact-terminal frame for both policies
and both cameras on floors 25, 26 and 31 against the physical-time source frame.
"""
    checks = []
    for case in cases:
        if case["floor"] not in (25, 26, 31):
            continue
        composite_path = case["results"][MAIN_VIDEO_KEY]["path"]
        for column, method in enumerate(METHODS):
            source = render_rows[(method, case["floor"])]
            end = source["sim_seconds"]
            for output_index in (round(end * 30 / 2), math.ceil(end * 30 - 1e-8)):
                output_time = output_index / 30
                source_index = max(0, bisect_right(source["frame_sim_seconds"], output_time + 1e-9) - 1)
                composed = decoded_frame(composite_path, output_index, 1280, 1152)
                for view_index, view in enumerate(("global", "wrist")):
                    measured = decoded_frame(source["videos"][view]["path"], source_index, 640, 480)
                    top, left = 112 + view_index * 480, column * 640
                    delta = np.abs(composed[top + 40:top + 470, left + 10:left + 630].astype(np.int16)
                                   - measured[40:470, 10:630].astype(np.int16))
                    error = float(delta.mean())
                    require(error < 5.0, f"Playback frame does not match measured time: floor {case['floor']} "
                            f"{method} {view} at {output_time:.3f}s, RGB MAE={error}")
                    checks.append(dict(floor=case["floor"], method=method, view=view,
                        output_time_seconds=output_time, source_frame=source_index,
                        source_physical_time_seconds=source["frame_sim_seconds"][source_index],
                        mean_absolute_rgb_error=error))
    require(len(checks) == 24, "Missing physical-time pixel checks")
    return checks


def composition():
    configure_plan()
    require(read(RUN / "render_qa.json")["status"] == "pass", "Render audit failed")
    render_manifest = RUN / "renders/render_manifest.json"
    rendered = read(render_manifest)
    record = read(RUN / "videos/composition_manifest.json")
    normalized = read(RUN / "videos/normalized_input.json")
    require(Path(record["source_manifest"]).resolve() == render_manifest.resolve(), "Different render manifest")
    require(record["source_manifest_sha256"] == sha(render_manifest), "Render manifest changed")
    compositor = Path(__file__).resolve().with_name("compose_rl_comparison_videos.py")
    require(record["script_sha256"] == sha(compositor), "Compositor source changed")
    require(record["methods"] == normalized["methods"] == list(METHODS), "Wrong comparison methods")
    require(record["fps"] == 30 and record["dimensions"] == [1280, 1152], "Wrong comparison format")
    require([case["floor"] for case in record["cases"]] == list(range(24, 36)), "Missing comparison tasks")
    require([case["floor"] for case in normalized["cases"]] == list(range(24, 36)), "Missing normalized tasks")
    render_rows = {(row["method"], row["floor"]): row for row in rendered["episodes"]}
    timeline = []
    cumulative_frames = 0
    for case, normalized_case in zip(record["cases"], normalized["cases"], strict=True):
        floor = case["floor"]
        require(case["methods"] == list(METHODS) and case["fps"] == 30, "Per-case methods or FPS changed")
        require(case["case_id"] == normalized_case["case_id"], "Normalized case order changed")
        terminal = max(render_rows[(method, floor)]["sim_seconds"] for method in METHODS)
        require(abs(case["common_terminal_seconds"] - terminal) < 1e-12, "Comparison motion retimed")
        require(case["end_hold_seconds"] == record["end_hold_seconds"], "Per-case end hold changed")
        expected_count = math.ceil(terminal * 30 - 1e-8) + round(case["end_hold_seconds"] * 30) + 1
        require(case["frame_count"] == expected_count, "Comparison duration does not preserve actual time")
        require(abs(case["duration_seconds"] - expected_count / 30) < 1e-12, "Case timing differs")
        for method in METHODS:
            source = render_rows[(method, floor)]
            normalized_source = normalized_case["methods"][method]
            require(normalized_source["frame_sim_seconds"] == source["frame_sim_seconds"],
                    "Composition changed measured frame timestamps")
            require(normalized_source["sim_seconds"] == source["sim_seconds"], "Composition changed episode duration")
            require(normalized_source["seed"] == source["seed"] == 20260942, "Composition mixed reset seeds")
            for field in ("success", "termination", "pressed_floors", "sim_seconds"):
                require(case["outcomes"][method][field] == source[field], "Composition changed measured outcome")
            for view in ("global", "wrist"):
                supplied = case["sources"][method][view]
                video = source["videos"][view]
                require(Path(supplied["path"]).resolve() == Path(video["path"]).resolve(), "Wrong source camera video")
                require(supplied["sha256"] == video["sha256"] == sha(video["path"]), "Source video changed")
                require(supplied["frame_count"] == source["frame_count"] and supplied["fps"] == 30,
                        "Source frame metadata changed")
        require(set(case["results"]) == {"comparison", *METHODS, MAIN_VIDEO_KEY},
                "Missing per-case video outputs")
        for name, video in case["results"].items():
            width = 640 if name in METHODS else 1280
            video_info(video["path"], expected_count, width, 1152)
        timeline.append(dict(floor=floor, start_seconds=cumulative_frames / 30,
                             duration_seconds=expected_count / 30,
                             physical_terminal_seconds={method: render_rows[(method, floor)]["sim_seconds"]
                                                        for method in METHODS}))
        cumulative_frames += expected_count
    require(set(record["collections"]) == {"comparison", *METHODS, MAIN_VIDEO_KEY},
            "Missing concatenated video outputs")
    for name, video in record["collections"].items():
        require(sha(video["path"]) == video["sha256"], "Concatenated video hash mismatch")
        video_info(video["path"], cumulative_frames, 640 if name in METHODS else 1280, 1152)
    pixel_checks = check_visible_frame_times(record["cases"], render_rows)
    final = record["collections"][MAIN_VIDEO_KEY]
    decode = subprocess.run([
        "ffmpeg", "-v", "error", "-xerror", "-nostdin", "-threads", "1", "-i", final["path"],
        "-map", "0:v:0", "-f", "null", "-"
    ], capture_output=True, text=True)
    require(decode.returncode == 0 and not decode.stderr.strip(), f"Full video decoding failed: {decode.stderr}")
    qa = dict(status="pass", video=final["path"], full_video_decode_exit_code=decode.returncode,
        decode_stderr=decode.stderr, frame_count=cumulative_frames, duration_seconds=cumulative_frames / 30,
        resolution=[1280, 1152], fps=30, bytes=Path(final["path"]).stat().st_size,
        sha256=final["sha256"], cases=12, methods=list(METHODS), main_video_key=MAIN_VIDEO_KEY,
        all_terminal_frames_retained=True,
        source_manifest_hash_matches=True, compositor_source_hash_matches=True,
        same_simulation_time_playback=True, speed_multiplier=1,
        physical_time_pixel_checks=pixel_checks,
        frame_mapping="Each output t=index/30 uses the latest measured source timestamp <=t; no interpolation. "
                      "The exact terminal state is held after termination; a partial final interval appears at the next 30 Hz slot.",
        timeline=timeline)
    write("composition_qa.json", qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path,
                        help="Right policy's completed 120-episode eval directory; default: plan evaluation_references/evaluation_reference, then legacy eval")
    parser.add_argument("--stage", choices=["trajectories", "renders", "composition", "all"], default="all")
    args = parser.parse_args()
    RUN = args.run.resolve(strict=True)
    REFERENCE = args.reference
    stages = {"trajectories": trajectories, "renders": renders, "composition": composition}
    for name in stages if args.stage == "all" else [args.stage]:
        stages[name]()
