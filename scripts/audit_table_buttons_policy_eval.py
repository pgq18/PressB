#!/usr/bin/env python3
"""Independently audit physical closed-loop table-button evaluation, on CPU.

Reconstruct every press and the full model-request -> units -> clamp -> applied
control -> four physics ticks chain. The reported success flag is never the
source of a success count. Incomplete/invalid episodes remain explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from audit_table_buttons_dataset import arm_limits, audit_arrays, audit_video, close, require
from pressb.tasks.table_buttons.spacing import resolve_spacing
from pressb.tasks.table_buttons.task import button_in_camera

COLORS = ("red", "yellow", "blue")
PROFILES = ("near", "medium", "far")
TO_SIM = np.array([np.pi / 180] * 6 + [.01])
SCALE = np.array([180.] * 6 + [8.])
JOINT6_SCHEMA = "vlajepa.table_buttons.joint6.v1"
FIXED_GRIPPER_FULL_WIDTH_M = float(np.float32(.02)) * .01
JOINT6_CONTRACT_KEYS = ("model_sha256", "best_step", "native_schema", "native_action_dim",
    "state_dim", "wire_action_dim", "action_horizon", "fixed_gripper_full_width_m", "gripper_learned")
JOINT6_MODELS = {"medium_red_joint6": ("medium", 100),
    **{f"{profile}_red_joint6_1000": (profile, 1000) for profile in PROFILES}}
JOINT6_MULTITASK_MODELS = {f"{profile}_multitask_joint6_3000": profile for profile in PROFILES}
JOINT6_MULTITASK_CONTRACT_KEYS = JOINT6_CONTRACT_KEYS + (
    "source_dataset_group", "source_episode_count", "task_type", "allowed_instructions")
THRESHOLDS = dict(press_depth_m=.003, release_depth_m=.001, min_contact_force_n=.1, hold_seconds=.025)
DUAL_CONTACT_POLICY = dict(version="table_buttons_dual_contact_v1",
    allowed_task_contact_pairs=[["/World/Piper/link7", "/World/Piper/link8"]],
    task_success="physical_target_press_without_wrong_or_blocking_contact",
    strict_collector_success="task_success_and_no_original_unexpected_contact")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=4096)
def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_arrays(directory):
    result = []
    for name in ("data", "physics"):
        with np.load(Path(directory) / f"{name}.npz", allow_pickle=False) as data:
            result.append({key: np.asarray(data[key]) for key in data.files})
    return result


def audit_timing(frames, physics, invalid_policy=False):
    if len(frames["sim_time"]) > 1:
        return audit_arrays(frames, physics)
    require(invalid_policy, "Zero-action episode must explicitly record invalid_policy_action")
    close(frames["sim_time"], [0.], "Zero-action frame time")
    close(physics["time_s"], [0.], "Zero-action physics time")
    for key in ("state", "action"):
        require(frames[key].shape == (1, 7) and np.isfinite(frames[key]).all(), f"Malformed zero-action {key}")
    for key, width in (("q_actual", 8), ("q_target", 8), ("travel_m", 3), ("finger_force_n", 3)):
        require(physics[key].shape == (1, width) and np.isfinite(physics[key]).all(), f"Malformed zero-action {key}")
    q = np.asarray(physics["q_actual"][0], dtype=np.float64)
    close(frames["state"][0], np.r_[q[:6], q[6]-q[7]], "Zero-action measured state mismatch")
    return 1, 1


def physical_outcome(physics, config, target, contacts, unexpected, collision_policy=None):
    """Independent hysteresis/dwell implementation; intentionally no Outcome call."""
    settings = config["success"]
    require(settings == THRESHOLDS, "Physical press thresholds differ from collection")
    times, travel, forces = (np.asarray(physics[key]) for key in ("time_s", "travel_m", "finger_force_n"))
    require(target in COLORS and len(times) > 0, "Invalid target or empty physics")
    require(travel.shape == forces.shape == (len(times), 3), "Invalid cap evidence shape")
    require(np.isfinite(travel).all() and np.isfinite(forces).all() and np.min(forces) >= 0, "Invalid cap evidence")
    require(np.max(np.abs(travel[0])) < settings["release_depth_m"] and np.max(forces[0]) <= .02,
            "Initial buttons not released/contact-free after settling")
    reconstructed = np.zeros_like(forces)
    for event in contacts:
        tick, color = event.get("tick"), event.get("color")
        require(type(tick) is int and 0 <= tick < len(times) and color in COLORS, "Invalid contact tick/color")
        require(event.get("fingers") and set(event["fingers"]) <= {"link7", "link8"}, "Invalid contact finger")
        force = float(event["force_n"])
        require(np.isfinite(force) and force >= 0, "Invalid contact force")
        reconstructed[tick, COLORS.index(color)] += force
    close(reconstructed, forces, "Contact log differs from per-tick forces", tolerance=1e-8)
    for event in unexpected:
        require(type(event.get("tick")) is int and 1 <= event["tick"] < len(times), "Invalid unexpected-contact tick")
        require(np.isfinite(event["force_n"]) and event["force_n"] > .02, "Unexpected contact must exceed 0.02 N")
        require(len(event.get("actors", [])) == 2 and event["actors"][0] != event["actors"][1], "Invalid collision actors")
        require(any(actor.startswith("/World/Piper/") and actor != "/World/Piper/base_link" for actor in event["actors"]),
                "Unexpected contact contains no moving robot link")
    require(collision_policy is None or collision_policy == DUAL_CONTACT_POLICY,
            "Unrecognized collision policy; only explicitly frozen finger closure may be allowed")
    finger_closure = [event for event in unexpected if set(event["actors"]) == {"/World/Piper/link7", "/World/Piper/link8"}]
    blocking = [event for event in unexpected if collision_policy is None
                or set(event["actors"]) != {"/World/Piper/link7", "/World/Piper/link8"}]
    events = []
    # A press needs elapsed dwell, not a count of four arbitrarily-spaced samples.
    for column, color in enumerate(COLORS):
        first_valid = None
        latched = False
        for index in range(1, len(times)):
            if travel[index, column] <= settings["release_depth_m"]:
                latched = False
            valid = travel[index, column] >= settings["press_depth_m"] and forces[index, column] >= settings["min_contact_force_n"]
            if not valid or latched:
                first_valid = None
            else:
                if first_valid is None:
                    first_valid = index
                if times[index] - times[first_valid] + 1e-12 >= settings["hold_seconds"]:
                    events.append(dict(tick=index, time_s=float(times[index]), color=color))
                    latched, first_valid = True, None
    events.sort(key=lambda event: (event["tick"], COLORS.index(event["color"])))
    target_events = [event for event in events if event["color"] == target]
    wrong_events = [event for event in events if event["color"] != target]
    other = [index for index, color in enumerate(COLORS) if color != target]
    wrong_ticks = np.flatnonzero(np.any(forces[:, other] > .02, axis=1)).tolist()
    firsts = [event["tick"] for event in events] + wrong_ticks
    strict_firsts = firsts + [event["tick"] for event in unexpected]
    task_firsts = firsts + [event["tick"] for event in blocking]
    task_success = bool(target_events and not wrong_events and not wrong_ticks and not blocking)
    strict_success = task_success and not unexpected
    return dict(success=task_success, task_success=task_success, strict_collector_success=bool(strict_success),
        target_pressed=bool(target_events), wrong_press=bool(wrong_events),
        non_target_contact=bool(wrong_ticks), unexpected_collision=bool(blocking),
        original_unexpected_contact=bool(unexpected), finger_closure_contact=bool(finger_closure),
        finger_closure_contact_count=len(finger_closure), blocking_contact_count=len(blocking),
        finger_closure_maximum_force_n=max((event["force_n"] for event in finger_closure), default=0.),
        first_strict_failure_time_s=min([event["time_s"] for event in wrong_events]
            + [tick/120 for tick in wrong_ticks] + [event["tick"]/120 for event in unexpected], default=None),
        first_success_time_s=target_events[0]["time_s"] if target_events else None,
        first_terminal_tick=min(task_firsts) if task_firsts else None,
        strict_first_terminal_tick=min(strict_firsts) if strict_firsts else None, press_events=events,
        non_target_contact_ticks=len(wrong_ticks), unexpected_contact_events=len(unexpected),
        maximum_travel_m=dict(zip(COLORS, np.max(travel, axis=0).tolist())),
        maximum_finger_force_n=dict(zip(COLORS, np.max(forces, axis=0).tolist())))


def audit_limits(report, config):
    limits = np.asarray(report["joint_limits_dof_rad_m"], dtype=float)
    require(limits.shape == (8, 2) and np.isfinite(limits).all() and np.all(limits[:, 0] < limits[:, 1]), "Malformed actual articulation limits")
    close(limits[:6], arm_limits(str((ROOT / config["robot_urdf"]).resolve())), "Physical arm limits differ from URDF", tolerance=2e-6)
    lower = np.r_[limits[:6, 0], max(0., 2*limits[6, 0], -2*limits[7, 1])]
    upper = np.r_[limits[:6, 1], min(2*limits[6, 1], -2*limits[7, 0])]
    require(np.all(lower < upper), "Inconsistent symmetric gripper bounds")
    require(abs(upper[6] - config["gripper_max_width_m"]) < 2e-7, "Physical gripper maximum differs from scene")
    close(report["commands"]["lower_rad_m"], lower, "Reported lower command bounds differ")
    close(report["commands"]["upper_rad_m"], upper, "Reported upper command bounds differ")
    return lower, upper


def audit_cameras_and_reset(report, frames, plan, cell):
    """Compare rendered sensor registration and settled initial state to collection."""
    path = Path(plan["source"]["source_collection_plans"][cell["spacing_profile"]]["path"])
    source = read_json(path.parent/f"{cell['spacing_profile']}_{cell['target_color']}"/"episode_000000"/"report.json")
    fields = ("resolution", "intrinsics", "optical_position_link6", "optical_quaternion_wxyz_link6",
              "source_sha256", "runtime_asset_sha256")
    for key in fields:
        require(report["wrist_camera"][key] == source["wrist_camera"][key], f"Wrist camera changed since collection: {key}")
    for name in ("up", "wrist"):
        require(report["cameras"][name]["resolution"] == source["cameras"][name]["resolution"] == [640, 480],
                f"Rendered {name} resolution changed")
        close(report["cameras"][name]["intrinsics"], source["cameras"][name]["intrinsics"],
              f"Rendered {name} intrinsics changed", tolerance=2e-4)
    close(report["cameras"]["up"]["world_from_camera_usd_at_initial"], source["cameras"]["up"]["world_from_camera_usd_at_initial"],
          "Fixed camera optical transform changed", tolerance=1e-6)
    q = np.asarray(source["initial_q_actual"])
    reference = np.r_[q[:6], q[6]-q[7]]
    difference = frames["state"][0]-reference
    require(np.max(np.abs(difference[:6])) < 5e-5 and abs(difference[6]) < 5e-6,
            "Settled initial measured state differs materially from collection")
    close(report["cameras"]["wrist"]["world_from_camera_usd_at_initial"], source["cameras"]["wrist"]["world_from_camera_usd_at_initial"],
          "Initial wrist optical transform changed", tolerance=1e-4)
    return dict(reference=str(path.parent/f"{cell['spacing_profile']}_{cell['target_color']}"/"episode_000000"/"report.json"),
        actual_sensor_registration_matches_collection=True, initial_measured_difference_rad_m=difference.tolist())


def audit_native_action_contract(model, cell):
    """Distinguish a six-output network from its seven-coordinate transport."""
    declared = cell.get("model_contract", {})
    joint6 = (model.get("native_schema") == JOINT6_SCHEMA or model.get("native_action_dim") == 6
              or declared.get("native_action_dim") == 6)
    if not joint6:
        return None  # Preserve the original joint7 protocol, including old logs.
    expected = dict(native_schema=JOINT6_SCHEMA, native_action_dim=6, state_dim=7,
        action_dim=7, wire_action_dim=7, wire_schema="pressb.table_buttons.joint7_policy.v1",
        action_horizon=7, noise_shape_per_sample=[1, 7, 6], noise_hash_array_shape=[7, 6],
        noise_hash_dtype="float32", noise_response_field="initial_noise", noise_hash_memory_order="C",
        native_action_units=["degree"] * 6,
        native_action_normalization_scale=[180.] * 6,
        fixed_gripper_full_width_m=FIXED_GRIPPER_FULL_WIDTH_M,
        fixed_gripper_full_width_cm=FIXED_GRIPPER_FULL_WIDTH_M * 100., gripper_learned=False)
    for key, value in expected.items():
        require(model.get(key) == value, f"Native joint6 model contract mismatch: {key}")
    require(model.get("model_id") == cell["model_id"]
            and cell["model_id"] in (JOINT6_MODELS | JOINT6_MULTITASK_MODELS),
            "Native joint6 model identity differs from selected model")
    if cell["model_id"] in JOINT6_MULTITASK_MODELS:
        profile = JOINT6_MULTITASK_MODELS[cell["model_id"]]
        require(cell.get("spacing_profile") == profile and cell.get("target_color") in COLORS
                and cell.get("model_type") == "multitask"
                and cell.get("instruction") == f"Press the {cell['target_color']} button.",
                "Native joint6 model differs from the selected multi-task instruction")
        for key, value in multitask_model_contract(profile).items():
            require(model.get(key) == value, f"Native joint6 model contract mismatch: {key}")
        require(all(key in declared for key in JOINT6_MULTITASK_CONTRACT_KEYS),
                "Incomplete frozen multi-task joint6 model contract")
    elif JOINT6_MODELS[cell["model_id"]][1] == 1000:
        profile, source_count = JOINT6_MODELS[cell["model_id"]]
        require(cell.get("spacing_profile") == profile and cell.get("target_color") == "red",
                "Native joint6 model differs from the selected task")
        for key, value in dict(source_dataset_group=f"{profile}_red", source_episode_count=source_count).items():
            require(model.get(key) == value, f"Native joint6 model contract mismatch: {key}")
    if declared:
        require(all(key in declared for key in JOINT6_CONTRACT_KEYS), "Incomplete frozen joint6 model contract")
        for key, value in declared.items():
            require(model.get(key) == value, f"Model differs from frozen joint6 contract: {key}")
    return expected


def audit_joint6_response(request, raw, normalized):
    """Check native actions/noise and the transport-only fixed gripper."""
    native = np.asarray(request.get("native_actions"), dtype=float)
    native_normalized = np.asarray(request.get("native_normalized_actions"), dtype=float)
    require(native.shape == native_normalized.shape == (7, 6)
            and np.isfinite(native).all() and np.isfinite(native_normalized).all(),
            "Missing or malformed native joint6 action evidence")
    require(np.array_equal(raw[:, :6], native) and np.array_equal(normalized[:, :6], native_normalized),
            "Native joint6 outputs changed while constructing the seven-coordinate wire")
    require(np.allclose(native, native_normalized * 180., atol=2e-5, rtol=2e-6),
            "Native joint6 inverse normalization differs from training")
    width_cm = FIXED_GRIPPER_FULL_WIDTH_M * 100.
    require(np.all(raw[:, 6] == width_cm) and np.all(normalized[:, 6] == width_cm / 8.),
            "Transport gripper is not the fixed dataset full-width command")
    noise = np.asarray(request.get("initial_noise"), dtype=np.float32)
    require(noise.shape == (7, 6) and np.isfinite(noise).all(),
            "Native initial_noise must contain the recorded float32 [7,6] sample, not joint7 noise")
    digest = hashlib.sha256(noise.tobytes()).hexdigest()
    require(request.get("initial_noise_sha256") == digest, "Native initial_noise bytes do not match service SHA256")
    # This validates recorded CUDA noise bytes and dimensions. It does not
    # recreate CUDA's seeded RNG with the different CPU generator algorithm.
    return digest


def audit_controls(frames, requests, controls, lower, upper, *, cell, row, model, invalid_policy=False):
    native_contract = audit_native_action_contract(model, cell)
    count = len(frames["sim_time"])
    require(len(controls) == count-1, "Each physical control interval needs exactly one model control record")
    expected_requests = (count-2)//7+1 if count > 1 else 0
    require(len(requests) == expected_requests + int(invalid_policy), "Incorrect request count for seven-action chunks")
    successful_requests = []
    for index, request in enumerate(requests):
        frame_index = index*7
        require(request["frame_index"] == frame_index and frame_index < count, "Request is not on a seven-control boundary")
        require(request["noise_seed"] == row["policy_seed"] + index, "Unpaired policy RNG seed")
        require(request["instruction"] == cell["instruction"], "Per-frame language differs from target instruction")
        close(request["state_deg_cm"], frames["state"][frame_index]/TO_SIM, "Inference state units/time mismatch", tolerance=2e-5)
        require(abs(request["sim_time_s"]-frame_index/30) < 1e-10, "Inference request simulation time mismatch")
        require(request["model"]["model_id"] == cell["model_id"] and request["model"]["model_sha256"] == model["model_sha256"], "Model identity changed during episode")
        if native_contract is not None:
            audit_native_action_contract(request["model"], cell)
            for key in JOINT6_CONTRACT_KEYS:
                require(request["model"].get(key) == model.get(key), f"Joint6 model contract changed during episode: {key}")
        if "error" in request:
            require(invalid_policy and index == len(requests)-1 and frame_index == count-1, "Error request is not terminal")
            continue
        if "normalized_state" in request:
            close(request["normalized_state"], np.asarray(request["state_deg_cm"])/SCALE,
                  "Service normalized measured state differently from training", tolerance=1e-7)
        raw, normalized = np.asarray(request["raw_actions_deg_cm"], dtype=float), np.asarray(request["normalized_actions"], dtype=float)
        require(raw.shape == normalized.shape == (7, 7) and np.isfinite(raw).all() and np.isfinite(normalized).all(), "Malformed/nonfinite successful model response")
        require(np.allclose(raw, normalized*SCALE, atol=2e-5, rtol=2e-6), "Action inverse normalization differs from training")
        if native_contract is not None:
            require("normalized_state" in request, "Joint6 response must retain measured seven-dimensional state evidence")
            audit_joint6_response(request, raw, normalized)
        successful_requests.append(request)
    require(len(successful_requests) == expected_requests, "Missing successful policy response")
    masks, differences = [], []
    for frame_index, control in enumerate(controls):
        request_index, action_index = divmod(frame_index, 7)
        request = successful_requests[request_index]
        require(control["frame_index"] == frame_index and control["request_id"] == request["request_id"] and control["action_index"] == action_index,
                "Executed command is not the corresponding model chunk action")
        raw = np.asarray(request["raw_actions_deg_cm"][action_index])
        converted = raw*TO_SIM
        clipped = np.clip(converted, lower, upper)
        close(control["raw_command_deg_cm"], raw, "Controller raw command differs from model")
        close(control["converted_rad_m"], converted, "Degrees/cm to radians/full metres conversion mismatch")
        close(control["clipped_rad_m"], clipped, "Controller clamp differs from physical joint bounds")
        close(control["applied_rad_m"], clipped, "Unexpected smoothing, IK, interpolation, or scripted action substitution")
        close(frames["action"][frame_index], clipped, "Recorded interval action differs from applied model command")
        mask = converted != clipped
        require(control["clipped_dimensions"] == np.flatnonzero(mask).tolist(), "Clipped dimension count mismatch")
        masks.append(mask)
        differences.append(np.abs(converted-clipped)/TO_SIM)
    masks = np.asarray(masks, dtype=bool).reshape(-1, 7)
    differences = np.asarray(differences).reshape(-1, 7)
    durations = [request["round_trip_seconds"] for request in successful_requests]
    require(all(np.isfinite(value) and value >= 0 for value in durations), "Invalid policy round-trip timing")
    result = dict(total_actions=len(controls), clipped_actions=int(np.any(masks, axis=1).sum()),
        clipping_action_fraction=float(np.any(masks, axis=1).mean()) if len(masks) else 0.,
        clipped_by_dimension=masks.sum(axis=0).tolist(),
        maximum_clamp_deg_cm=differences.max(axis=0).tolist() if len(differences) else [0.]*7,
        policy_requests=len(requests), successful_policy_requests=len(successful_requests),
        inference_round_trip_seconds=durations)
    if native_contract is not None:
        result["native_action_audit"] = dict(native_action_dim=6, wire_action_dim=7, state_dim=7,
            noise_shape_per_sample=[1, 7, 6], noise_hash_array_shape=[7, 6],
            verified_native_noise_samples=len(successful_requests),
            fixed_gripper_full_width_m=FIXED_GRIPPER_FULL_WIDTH_M, gripper_learned=False,
            cuda_rng_reconstructed_on_cpu=False)
    return result


def failure_reason(outcome, invalid_policy):
    if outcome["unexpected_collision"]:
        return "unexpected_contact"
    if outcome["non_target_contact"]:
        return "non_target_finger_contact"
    if outcome["wrong_press"]:
        return "wrong_button"
    if invalid_policy:
        return "invalid_policy_action"
    return "success" if outcome["success"] else "timeout"


def audit_episode(directory, cell, row, plan, *, decode_videos=False, plan_sha256=None):
    directory = Path(directory)
    metadata, report, cfg = [read_json(directory/f"{name}.json") for name in ("metadata", "report", "config")]
    require(metadata["schema_version"] == "pressb.table_buttons.policy_eval.v1", "Unknown raw evaluation schema")
    for value in (metadata, report):
        for key, expected in (("cell_id", cell["cell_id"]), ("model_id", cell["model_id"]),
                              ("spacing_profile", cell["spacing_profile"]), ("target_color", cell["target_color"]),
                              ("seed", row["seed"]), ("instruction", cell["instruction"])):
            require(value[key] == expected, f"Episode identity mismatch: {key}")
        require(type(value["success"]) is bool, "Reported success must be boolean")
    require(report["evaluation_status"] == "complete", "Episode output is incomplete")
    require(report["policy_or_RL_evaluation"] is True, "Not a learned-policy rollout")
    require(report["expert_actions_used"] is False and metadata["expert_actions_used"] is False,
            "Expert control reported during learned-policy rollout")
    require(report["network_wait_advances_simulation"] is False, "Network latency advances simulation")
    require(metadata["fps"] == 30 and abs(metadata["physics_dt"]-1/120) < 1e-12,
            "Metadata physical/control rates differ")
    require(metadata["state_units"] == metadata["action_units"] == ["rad"]*6+["m"]
            and metadata["gripper_representation"] == "full_width", "Raw units differ from joint7 contract")
    require(metadata["policy_normalization_scale"] == SCALE.tolist(), "Policy normalization metadata differs")
    require(cfg == read_json(row["effective_config_file"]), "Actual scene differs from frozen per-layout configuration")
    require(report["config"] == cfg, "Rendered scene report configuration differs")
    require(metadata["config_sha256"] == canonical_sha(cfg), "Actual scene canonical configuration hash differs")
    require(metadata["layout_sha256"] == row["layout_sha256"], "Actual layout file hash differs from frozen layout")
    if plan_sha256 is not None:
        require(metadata["plan_sha256"] == plan_sha256, "Runtime used a different plan file")
    require(cfg["success"] == THRESHOLDS, "Wrong physical success thresholds")
    frames, physics = load_arrays(directory)
    cameras_and_reset = audit_cameras_and_reset(report, frames, plan, cell) if "source" in plan else None
    invalid_policy = report["stop_reason"] == "invalid_policy_action"
    count, ticks = audit_timing(frames, physics, invalid_policy)
    duration = (count-1)/30
    require(duration <= plan["defaults"]["timeout_s"]+1e-9, "Episode exceeds physical timeout")
    close(metadata["initial_state_measured_rad_m"], frames["state"][0], "Reported initial measurement differs")
    target = np.r_[cfg["initial_joint_positions_rad"], cfg["initial_gripper_width_m"]]
    close(physics["q_target"][0], np.r_[target[:6], target[6]/2, -target[6]/2], "Reset does not use training initial target")
    unexpected = read_json(directory/"unexpected_contacts.json")
    collision_policy = plan["defaults"].get("collision_policy")
    outcome = physical_outcome(physics, cfg, cell["target_color"], read_json(directory/"contacts.json"), unexpected, collision_policy)
    if collision_policy is not None:
        closure = [event for event in unexpected if set(event["actors"]) == {"/World/Piper/link7", "/World/Piper/link8"}]
        blocking = [event for event in unexpected if set(event["actors"]) != {"/World/Piper/link7", "/World/Piper/link8"}]
        require(read_json(directory/"finger_closure_contacts.json") == closure, "Finger closure split differs from original contact log")
        require(read_json(directory/"blocking_contacts.json") == blocking, "Blocking collision split differs from original contact log")
        for value in (metadata, report):
            require(value["collision_policy"] == collision_policy, "Runtime contact policy differs from frozen plan")
            require(value["task_success"] == outcome["task_success"] and value["strict_collector_success"] == outcome["strict_collector_success"],
                    "Reported dual success metrics disagree with physical evidence")
        require(report["finger_closure_contact_count"] == len(closure) and report["blocking_contact_count"] == len(blocking),
                "Reported contact classification counts disagree")
    if outcome["first_terminal_tick"] is not None:
        require(count-1 == math.ceil(outcome["first_terminal_tick"]/4), "Episode did not end at the first control boundary after physical termination")
    elif not invalid_policy:
        require(abs(duration-plan["defaults"]["timeout_s"]) < 1e-9, "Nonterminal policy episode ended before timeout")
    for key in ("success", "task_success", "strict_collector_success"):
        outcome[key] = outcome[key] and not invalid_policy
    reason = failure_reason(outcome, invalid_policy)
    for value in (metadata, report):
        require(value["success"] == outcome["success"], "Reported success disagrees with independent physical outcome")
        require(value["stop_reason"] == reason, "Reported stop reason differs from raw physical evidence")
        require(abs(value["duration_s"]-duration) < 1e-10, "Reported duration differs from physical timeline")
    lower, upper = audit_limits(report, cfg)
    model = report["model"]
    require(metadata["model_sha256"] == model["model_sha256"] and metadata["model"] == model,
            "Metadata model identity differs from report")
    require(model["strict_loaded"] is True and model["model_digest_verified"] is True,
            "Service did not confirm strict verified checkpoint loading")
    require(model["camera_order"] == ["up", "wrist"] and model["image_input_size"] == [640, 480]
            and model["image_model_size"] == [224, 224], "Model camera order/resolution differs from training")
    require(model["normalization_scale"] == SCALE.tolist() and model["action_horizon"] == model["action_dim"] == 7,
            "Model action schema differs from training")
    commands = audit_controls(frames, read_lines(directory/"policy_requests.jsonl"), read_lines(directory/"control.jsonl"),
        lower, upper, cell=cell, row=row, model=model, invalid_policy=invalid_policy)
    for key in ("total_actions", "clipped_actions", "clipped_by_dimension"):
        require(commands[key] == report["commands"][key], f"Controller summary {key} differs from independently counted evidence")
    require(report["commands"]["smoothing"]["mode"] == "none", "Formal evaluation unexpectedly uses smoothing")
    require(np.all(physics["q_target"][:, :6] >= lower[:6]-1e-8) and np.all(physics["q_target"][:, :6] <= upper[:6]+1e-8), "Applied arm target exceeds physical bounds")
    require(np.all(frames["action"][:, 6] >= lower[6]-1e-10) and np.all(frames["action"][:, 6] <= upper[6]+1e-10), "Applied gripper target exceeds physical full-width bounds")
    videos = {}
    if decode_videos:
        for camera in ("up", "wrist"):
            if (directory/f"{camera}.mp4").is_file():
                videos[camera] = audit_video(directory/f"{camera}.mp4", count)
    return dict(audit_valid=True, directory=str(directory), cell_id=cell["cell_id"], model_id=cell["model_id"],
        model_sha256=model["model_sha256"], spacing_profile=cell["spacing_profile"], target_color=cell["target_color"],
        episode_index=row["episode_index"], seed=row["seed"], policy_seed=row["policy_seed"],
        positions_xy_m={color: cfg["buttons"][color]["center_xy"] for color in COLORS},
        layout_sha256=row["layout_sha256"], effective_config_sha256=row["effective_config_sha256"],
        duration_s=duration, frames=count, physics_samples=ticks, stop_reason=reason,
        invalid_policy_action=invalid_policy, initial_tracking_error_rad_m=(frames["state"][0]-target).tolist(),
        maximum_arm_tracking_error_rad=float(np.max(np.abs(physics["q_actual"][:, :6]-physics["q_target"][:, :6]))),
        **outcome, commands=commands, videos=videos, cameras_and_reset=cameras_and_reset)


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return None
    proportion = successes/total
    divisor = 1+z*z/total
    middle = (proportion+z*z/(2*total))/divisor
    half = z*math.sqrt(proportion*(1-proportion)/total+z*z/(4*total*total))/divisor
    return [max(0., middle-half), min(1., middle+half)]


def summary(records, planned, missing, invalid):
    successes = sum(bool(row["success"]) for row in records)
    strict_successes = sum(bool(row.get("strict_collector_success", row["success"])) for row in records)
    completed = len(records)+len(invalid)
    categories = Counter(row["stop_reason"] for row in records)
    success_times = [row["first_success_time_s"] for row in records if row["success"]]
    actions = sum(row["commands"]["total_actions"] for row in records)
    clipped = sum(row["commands"]["clipped_actions"] for row in records)
    latency = [value for row in records for value in row["commands"]["inference_round_trip_seconds"]]
    return dict(planned=planned, completed=completed, audited=len(records), missing=len(missing), invalid=len(invalid),
        successes=successes, audited_failures=len(records)-successes, complete=completed==planned,
        task_successes=successes, strict_collector_successes=strict_successes,
        success_rate=successes/completed if completed == planned and completed else None,
        success_rate_over_completed=successes/completed if completed else None,
        strict_collector_success_rate=strict_successes/completed if completed == planned and completed else None,
        strict_collector_success_rate_over_completed=strict_successes/completed if completed else None,
        strict_collector_success_rate_wilson_95=wilson(strict_successes, completed),
        episodes_with_finger_closure_contact=sum(bool(row.get("finger_closure_contact")) for row in records),
        success_rate_note="Invalid completed episodes count as non-success; missing episodes are reported separately.",
        success_rate_wilson_95=wilson(successes, completed), success_rate_planned_lower_bound=successes/planned if planned else None,
        stop_reasons=dict(categories), failure_flags={key: sum(bool(row[key]) for row in records) for key in
            ("wrong_press", "non_target_contact", "unexpected_collision", "invalid_policy_action")},
        success_time_mean_s=float(np.mean(success_times)) if success_times else None,
        success_time_median_s=float(np.median(success_times)) if success_times else None,
        actions=actions, clipped_actions=clipped, clipping_action_fraction=clipped/actions if actions else 0.,
        episodes_with_clipping=sum(row["commands"]["clipped_actions"]>0 for row in records),
        inference_round_trip_mean_s=float(np.mean(latency)) if latency else None,
        inference_round_trip_p95_s=float(np.percentile(latency, 95)) if latency else None)


def audit_joint6_plan_scope(plan):
    """Allow exactly the declared single-model comparison, not arbitrary subsets."""
    scope = plan["evaluation_scope"]
    model_id = scope.get("model_id")
    require(scope.get("kind") == "joint6_single_model" and model_id in JOINT6_MODELS,
            "Unknown evaluation scope")
    profile, source_count = JOINT6_MODELS[model_id]
    dataset_group = f"{profile}_red"
    reference_path = scope["reference_plan_file"]
    require(file_sha(reference_path) == scope["reference_plan_sha256"], "Joint6 reference plan changed")
    reference = read_json(reference_path)
    require(reference.get("schema") == "pressb.table_buttons.policy_eval_plan.v1"
            and "evaluation_scope" not in reference, "Joint6 reference must be the original joint7 evaluation plan")
    require(scope["reference_cell_id"] == dataset_group,
            f"Joint6 comparison must use original {dataset_group} layouts")
    references = [cell for cell in reference["cells"] if cell["cell_id"] == scope["reference_cell_id"]]
    require(len(references) == 1, "Joint6 reference cell is absent or ambiguous")
    original = references[0]
    require(len(plan["cells"]) == 1, "Joint6 comparison must contain exactly one model/task cell")
    cell = plan["cells"][0]
    for key, value in dict(cell_id=model_id, model_id=model_id, model_type="single",
                           spacing_profile=profile, target_color="red", instruction="Press the red button.").items():
        require(cell.get(key) == value, f"Joint6 evaluation cell mismatch: {key}")
    require(plan["config"] == reference["config"] and plan["source"] == reference["source"],
            "Joint6 comparison changed the original scene/source configuration")
    require(plan["defaults"] == reference["defaults"] and plan["defaults"].get("collision_policy") == DUAL_CONTACT_POLICY,
            "Joint6 comparison must preserve the original dual-metric evaluation defaults")
    require(len(cell["episodes"]) == 100 and cell["episodes"] == original["episodes"],
            f"Joint6 evaluation must reuse all 100 original {dataset_group} episode rows without changes")
    require(plan["planned_episodes"] == 100, "Joint6 comparison denominator must be 100 episodes")
    contract = cell.get("model_contract", {})
    require(all(key in contract for key in JOINT6_CONTRACT_KEYS), "Incomplete frozen joint6 model contract")
    for key, value in dict(native_schema=JOINT6_SCHEMA, native_action_dim=6, state_dim=7, wire_action_dim=7,
                           action_horizon=7, fixed_gripper_full_width_m=FIXED_GRIPPER_FULL_WIDTH_M,
                           gripper_learned=False).items():
        require(contract.get(key) == value, f"Frozen joint6 model contract mismatch: {key}")
    if source_count == 1000:
        for key, value in dict(source_dataset_group=dataset_group, source_episode_count=source_count).items():
            require(contract.get(key) == value, f"Frozen joint6 model contract mismatch: {key}")
    digest = contract["model_sha256"]
    require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
            "Frozen joint6 checkpoint SHA256 is invalid")
    require(type(contract["best_step"]) is int and contract["best_step"] > 0, "Frozen joint6 checkpoint step is invalid")
    return scope


def multitask_model_contract(profile):
    return dict(source_dataset_group=f"{profile}_multitask", source_episode_count=3000,
                task_type="multitask", allowed_instructions=[f"Press the {color} button." for color in COLORS])


def audit_joint6_multitask_plan_scope(plan):
    """Tie all three language tasks to one checkpoint and the original test scenes."""
    scope = plan["evaluation_scope"]
    model_id = scope.get("model_id")
    require(scope.get("kind") == "joint6_multitask_model" and model_id in JOINT6_MULTITASK_MODELS,
            "Unknown multi-task joint6 evaluation scope")
    profile = JOINT6_MULTITASK_MODELS[model_id]
    reference_path = scope["reference_plan_file"]
    require(file_sha(reference_path) == scope["reference_plan_sha256"], "Joint6 reference plan changed")
    reference = read_json(reference_path)
    require(reference.get("schema") == "pressb.table_buttons.policy_eval_plan.v1"
            and "evaluation_scope" not in reference, "Joint6 reference must be the original joint7 evaluation plan")
    reference_cells = {color: f"{profile}_multitask_{color}" for color in COLORS}
    require(scope.get("reference_cells") == reference_cells,
            "Multi-task comparison must use the original spacing/color reference cells")
    require(plan["config"] == reference["config"] and plan["source"] == reference["source"],
            "Joint6 comparison changed the original scene/source configuration")
    require(plan["defaults"] == reference["defaults"] and plan["defaults"].get("collision_policy") == DUAL_CONTACT_POLICY,
            "Joint6 comparison must preserve the original dual-metric evaluation defaults")
    require(len(plan["cells"]) == 3 and {cell.get("target_color") for cell in plan["cells"]} == set(COLORS),
            "Multi-task comparison must contain exactly three distinct color cells")
    require(plan["planned_episodes"] == 300 and plan.get("models") == 1 and plan.get("cells_count") == 3,
            "Multi-task comparison requires one model, three cells and 300 episodes")
    first_contract = None
    for cell in plan["cells"]:
        color = cell["target_color"]
        for key, value in dict(cell_id=f"{model_id}_{color}", model_id=model_id, model_type="multitask",
                               spacing_profile=profile, instruction=f"Press the {color} button.").items():
            require(cell.get(key) == value, f"Joint6 multi-task evaluation cell mismatch: {key}")
        originals = [entry for entry in reference["cells"] if entry["cell_id"] == reference_cells[color]]
        require(len(originals) == 1, "Joint6 reference cell is absent or ambiguous")
        require(len(cell["episodes"]) == 100 and cell["episodes"] == originals[0]["episodes"],
                f"Joint6 evaluation must reuse all 100 original {reference_cells[color]} episode rows without changes")
        contract = cell.get("model_contract", {})
        require(all(key in contract for key in JOINT6_MULTITASK_CONTRACT_KEYS),
                "Incomplete frozen multi-task joint6 model contract")
        expected = dict(native_schema=JOINT6_SCHEMA, native_action_dim=6, state_dim=7, wire_action_dim=7,
                        action_horizon=7, fixed_gripper_full_width_m=FIXED_GRIPPER_FULL_WIDTH_M,
                        gripper_learned=False, **multitask_model_contract(profile))
        for key, value in expected.items():
            require(contract.get(key) == value, f"Frozen joint6 model contract mismatch: {key}")
        digest = contract["model_sha256"]
        require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                "Frozen joint6 checkpoint SHA256 is invalid")
        require(type(contract["best_step"]) is int and contract["best_step"] > 0,
                "Frozen joint6 checkpoint step is invalid")
        if first_contract is None:
            first_contract = contract
        require(contract == first_contract, "All three tasks must use the same frozen model contract")
    return scope


def audit_plan(plan):
    require(plan["schema"] == "pressb.table_buttons.policy_eval_plan.v1", "Wrong suite plan schema")
    require(canonical_sha(plan["config"]) == plan["source"]["original_config_sha256"], "Frozen configuration fingerprint mismatch")
    source_domains = {}
    for profile, source in plan["source"]["source_collection_plans"].items():
        require(file_sha(source["path"]) == source["sha256"], "Source collection plan changed")
        require(read_json(source["path"])["config"] == plan["config"], "Source collection configuration differs")
        first_scene = read_json(Path(source["path"]).parent/f"{profile}_red"/"episode_000000"/"config.json")
        source_domains[profile] = first_scene["sampling_domain"]
    default = plan["defaults"]
    require(default["fps"] == 30 and abs(default["physics_dt"]-1/120)<1e-12 and default["execute_steps"] == 7,
            "Formal suite must preserve 30 Hz / 120 Hz / seven-action chunks")
    require(default["smoothing"]["mode"] == "none" and default["timeout_s"] == 12., "Formal suite requires no smoothing and 12 s timeout")
    require(default.get("collision_policy") is None or default["collision_policy"] == DUAL_CONTACT_POLICY,
            "Suite declares a different collision exception")
    scope = None
    if "evaluation_scope" in plan:
        scope = (audit_joint6_multitask_plan_scope(plan)
                 if plan["evaluation_scope"].get("kind") == "joint6_multitask_model"
                 else audit_joint6_plan_scope(plan))
    if scope is None:
        expected_cells = {(profile, kind, color) for profile in PROFILES for kind in ("single", "multitask") for color in COLORS}
        require({(c["spacing_profile"], c["model_type"], c["target_color"]) for c in plan["cells"]} == expected_cells and len(plan["cells"]) == 18,
                "Suite must contain all 18 independent model/task cells")
    shared_layouts, shared_scenes, shared_rng = {}, {}, {}
    for cell in plan["cells"]:
        profile, color = cell["spacing_profile"], cell["target_color"]
        require(cell["instruction"] == f"Press the {color} button.", "Task instruction differs across datasets/evaluation")
        require([row["episode_index"] for row in cell["episodes"]] == list(range(len(cell["episodes"]))), "Episode IDs not contiguous")
        for row in cell["episodes"]:
            require(row["seed"] == 10000+row["episode_index"], "Layout seed differs from unseen paired test protocol")
            for prefix in ("layout", "effective_config", "preflight"):
                require(file_sha(row[prefix+"_file"]) == row[prefix+"_sha256"], f"Frozen {prefix} artifact changed")
            layout = read_json(row["layout_file"])
            cfg = read_json(row["effective_config_file"])
            original_domain = source_domains[profile]
            fixed_domain = {key: value for key, value in original_domain.items() if key not in ("sampling", "layout_clearance")}
            require({key: value for key, value in layout["domain"].items() if key not in ("sampling", "layout_clearance")} == fixed_domain,
                    "Sampling camera/reachable domain differs from collected data")
            for key in ("proposal", "exact_candidate_ik", "discrete_path_collision_check", "complete_button_layout_check",
                        "spacing_profile", "min_separation_m", "max_separation_m"):
                require(layout["domain"]["sampling"][key] == original_domain["sampling"][key],
                        f"Sampling rule differs from collection: {key}")
            expected = deepcopy(plan["config"])
            expected["position_sampling"]["spacing_profile"] = profile
            expected["task_color"], expected["task_instruction"] = color, cell["instruction"]
            expected["press_approaches"], expected["sampling_domain"] = layout["approaches"], layout["domain"]
            for name in COLORS:
                expected["buttons"][name]["center_xy"] = layout["positions"][name]
            require(expected == cfg, "Effective scene changes more than layout and language target")
            key = profile, row["seed"]
            require(shared_layouts.setdefault(key, row["layout_sha256"]) == row["layout_sha256"], "Layouts are not paired across models/colors")
            key = profile, color, row["seed"]
            require(shared_scenes.setdefault(key, row["effective_config_sha256"]) == row["effective_config_sha256"], "Single/multitask scenes differ")
            require(shared_rng.setdefault(key, row["policy_seed"]) == row["policy_seed"], "Single/multitask policy noise is unpaired")
            spacing = resolve_spacing(cfg["position_sampling"], cfg["button_geometry"]["base_radius_m"])
            for index, name in enumerate(COLORS):
                xy = np.asarray(layout["positions"][name])
                require(button_in_camera(xy, cfg["up_camera"], button_radius=cfg["button_geometry"]["base_radius_m"],
                    button_top_z=cfg["button_geometry"]["base_height_m"]+cfg["button_geometry"]["cap_height_m"],
                    margin_px=cfg["position_sampling"]["pixel_margin"]), "Button outside camera visibility region")
                for other in COLORS[index+1:]:
                    require(spacing.accepts(float(np.linalg.norm(xy-np.asarray(layout["positions"][other])))), "Layout violates spacing band")
    require(sum(len(cell["episodes"]) for cell in plan["cells"]) == plan["planned_episodes"], "Planned denominator differs from cells")
    result = dict(valid=True, cells=len(plan["cells"]), episodes=plan["planned_episodes"], distinct_layouts=len(shared_layouts),
        original_config_sha256=plan["source"]["original_config_sha256"], collision_policy=default.get("collision_policy"))
    if scope is not None:
        result["evaluation_scope"] = scope
    return result


def audit_suite(suite_root, *, allow_partial=False, decode_videos=False):
    suite_root = Path(suite_root).resolve()
    plan = read_json(suite_root/"plan.json")
    plan_result = audit_plan(plan)
    records, missing, invalid, cells = [], [], [], {}
    for cell in plan["cells"]:
        own_records, own_missing, own_invalid = [], [], []
        for row in cell["episodes"]:
            directory = suite_root/"cells"/cell["cell_id"]/"episodes"/f"episode_{row['episode_index']:06d}"
            entry = dict(cell_id=cell["cell_id"], episode_index=row["episode_index"], seed=row["seed"], directory=str(directory))
            if not (directory/"report.json").is_file():
                own_missing.append(entry)
                continue
            try:
                own_records.append(audit_episode(directory, cell, row, plan, decode_videos=decode_videos,
                    plan_sha256=file_sha(str(suite_root/"plan.json"))))
            except Exception as error:
                own_invalid.append(dict(entry, error=f"{type(error).__name__}: {error}"))
        cells[cell["cell_id"]] = dict(model_id=cell["model_id"], model_type=cell["model_type"],
            spacing_profile=cell["spacing_profile"], target_color=cell["target_color"],
            **summary(own_records, len(cell["episodes"]), own_missing, own_invalid))
        records.extend(own_records); missing.extend(own_missing); invalid.extend(own_invalid)
    identities, identity_errors = {}, []
    for record in records:
        key, sha = record["model_id"], record["model_sha256"]
        if identities.setdefault(key, sha) != sha:
            identity_errors.append(dict(cell_id=record["cell_id"], episode_index=record["episode_index"], error="Model checkpoint changed during suite"))
    return dict(schema="pressb.table_buttons.policy_eval_audit.v1", created_at=datetime.now(timezone.utc).isoformat(),
        suite_root=str(suite_root), audit_passed=not invalid and not identity_errors and (allow_partial or not missing),
        complete=not missing, allow_partial=allow_partial, independent_physical_success=True,
        plan=plan_result, overall=summary(records, plan["planned_episodes"], missing, invalid),
        cells=cells, paired_single_vs_multitask=paired_comparisons(records),
        models=identities, episodes=records, invalid=invalid, missing=missing,
        identity_errors=identity_errors)


def paired_comparisons(records):
    """Report matched outcomes without treating shared layouts as independent."""
    index = {(row["cell_id"], row["seed"]): row for row in records}
    result = {}
    for profile in PROFILES:
        for color in COLORS:
            single_id, multi_id = f"{profile}_{color}", f"{profile}_multitask_{color}"
            pairs = [(a, index[(multi_id, seed)]) for (cell, seed), a in index.items()
                     if cell == single_id and (multi_id, seed) in index]
            both = sum(a["success"] and b["success"] for a, b in pairs)
            neither = sum(not a["success"] and not b["success"] for a, b in pairs)
            single_only = sum(a["success"] and not b["success"] for a, b in pairs)
            multi_only = sum(not a["success"] and b["success"] for a, b in pairs)
            discordant = single_only+multi_only
            p = min(1., 2*sum(math.comb(discordant, k) for k in range(min(single_only, multi_only)+1))/2**discordant) if discordant else 1.
            result[single_id] = dict(paired_audited_episodes=len(pairs), both_success=both, both_failure=neither,
                single_only_success=single_only, multitask_only_success=multi_only,
                multitask_minus_single_success_rate=(multi_only-single_only)/len(pairs) if pairs else None,
                mcnemar_exact_two_sided_p=p if pairs else None,
                note="Only matching audited layout seeds; incomplete pairs are omitted from this paired comparison, not from suite coverage.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--decode-videos", action="store_true")
    parser.add_argument("--episode-dir", type=Path, help="Audit a smoke episode outside cells/, using this suite's frozen plan")
    parser.add_argument("--cell", help="Cell ID for --episode-dir")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.episode_dir) != bool(args.cell):
        parser.error("--episode-dir and --cell are required together")
    output = args.output or args.suite_root/"audit"/(f"{args.cell}_{args.episode_index:06d}.json" if args.episode_dir else "summary.json")
    try:
        if args.episode_dir:
            plan = read_json(args.suite_root/"plan.json")
            audit_plan(plan)
            cell = next(cell for cell in plan["cells"] if cell["cell_id"] == args.cell)
            row = next(row for row in cell["episodes"] if row["episode_index"] == args.episode_index)
            result = dict(audit_passed=True, episode=audit_episode(args.episode_dir, cell, row, plan,
                decode_videos=args.decode_videos, plan_sha256=file_sha(str(args.suite_root/"plan.json"))))
        else:
            result = audit_suite(args.suite_root, allow_partial=args.allow_partial, decode_videos=args.decode_videos)
    except Exception as error:
        result = dict(audit_passed=False, error=f"{type(error).__name__}: {error}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    temporary.replace(output)
    print(json.dumps(dict(audit_passed=result["audit_passed"], report=str(output), overall=result.get("overall"), error=result.get("error"))))
    return 0 if result["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
