#!/usr/bin/env python3
"""Offline audit of saved PAP trials. Does not execute policy or simulation.

The full default audit writes rollout_audit.json. --limit is useful for a small
preflight; it writes rollout_audit_sample.json and never claims full completion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
from pressb.pick_place.task import PickPlaceOutcome
from pressb.pick_place.wallx import unpack


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def max_error(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return float(np.max(np.abs(left-right))) if left.size else 0.


def audit_episode(directory, episode, baseline_config, baseline_metadata):
    report = read(directory/"report.json")
    config = read(directory/"config.json")
    scene, policy = report["scene"], report["policy"]
    issues, measures = [], {}

    def check(condition, name, **details):
        if not condition:
            issues.append(dict(check=name, **details))

    check(report["evaluation_episode"] == episode, "episode_matches_plan")
    check(report["evaluation_status"] == "complete", "report_complete")
    check(config == report["config"], "saved_config_matches_report")
    check(policy["endpoint"] == baseline_metadata["endpoint"] and
          policy["server_metadata"] == baseline_metadata["server_metadata"], "server_metadata_and_endpoint_unchanged")
    check(policy["execute_steps"] == 32 and policy["predicted_horizon"] == 32 and policy["control_hz"] == 30,
          "policy_horizon_and_frequency")
    check(policy["arm_limit_mode"] == "error" and policy["gripper_limit_mode"] == "clip", "limit_modes")
    check(not policy["expert_actions_used"] and not report["expert_actions_used"] and
          not report["object_attachment_used"] and not report["object_teleport_after_initialization"], "no_expert_or_attachment")
    check(policy["smoothing"]["mode"] == "joint_ema" and policy["smoothing"]["alpha"] == .8,
          "smoothing_configuration")
    object_spec = config.get("object", config.get("cube"))
    check(object_spec["edge"] == .03 and object_spec["mass"] == .025 and object_spec["yaw_deg"] == 0.,
          "fixed_cube_parameters")
    check(scene["object_geometry"]["shape"] == "cube" and scene["object_geometry"]["size_xyz"] == [.03]*3
          and scene["object_mass"] == .025, "authored_cube_parameters")
    check(scene["appearance"]["object_material"]["color"] == episode["color_rgb"]
          and config["materials"]["RedCube"]["color"] == episode["color_rgb"], "authored_color_matches_plan")
    base_object_material = baseline_config["materials"]["RedCube"]
    for key in ("roughness", "metallic", "specular"):
        if key in base_object_material:
            check(scene["appearance"]["object_material"][key] == base_object_material[key], f"material_{key}_unchanged")
    # Configuration equality excludes only the two intended randomized fields.
    normalized_config = json.loads(json.dumps(config))
    object_key = "object" if "object" in config else "cube"
    normalized_config[object_key]["center_xy"] = baseline_config[object_key]["center_xy"]
    if object_key == "object" and "color" in baseline_config[object_key]:
        normalized_config[object_key]["color"] = baseline_config[object_key]["color"]
    if "color" in baseline_config["materials"]["RedCube"]:
        normalized_config["materials"]["RedCube"]["color"] = baseline_config["materials"]["RedCube"]["color"]
    else:
        normalized_config["materials"]["RedCube"].pop("color", None)
    check(normalized_config == baseline_config, "configuration_changes_only_xy_and_color")
    for name in ("up", "wrist"):
        spec = baseline_config[name+"_camera"]
        expected_k = [[spec["fx"], 0., spec["cx"]], [0., spec["fy"], spec["cy"]], [0., 0., 1.]]
        error = max_error(report["cameras"][name]["intrinsics"], expected_k)
        measures[name+"_intrinsics_max_error_px"] = error
        check(error <= 1e-4 and report["cameras"][name]["resolution"] == spec["resolution"], f"{name}_camera_intrinsics")

    with np.load(directory/"physics.npz", allow_pickle=False) as archive:
        physics = {key: archive[key] for key in archive.files}
    n = len(physics["time_s"])
    dt = baseline_config["physics_dt"]
    check(n == report["physics_samples"] and n > 0, "physics_sample_count")
    check(all(np.isfinite(values).all() for values in physics.values()), "finite_physics_records")
    check(max_error(physics["time_s"], np.arange(n)*dt) < 1e-12, "continuous_120hz_physics_time")
    check(report["duration_s"] == float(physics["time_s"][-1]) and report["duration_s"] <= 20., "simulation_duration")
    base_xy = np.array(scene["robot_base_position_world"][:2])
    expected_xy = np.array(episode["center_xy"])+base_xy
    measures["initial_xy_max_error_m"] = max_error(physics["cube_position"][0, :2], expected_xy)
    check(measures["initial_xy_max_error_m"] <= 1e-5, "settled_initial_xy_matches_plan")
    check(max_error(scene["object_initial_position_world"][:2], expected_xy) <= 1e-12, "authored_initial_xy_matches_plan")
    measures["initial_rotation_vector_norm_rad"] = float(2*np.linalg.norm(physics["cube_quaternion"][0, 1:]))
    check(measures["initial_rotation_vector_norm_rad"] < 1e-3, "settled_initial_cube_yaw_zero")
    check(np.array_equal(physics["q_actual"][0], np.asarray(report["initial_q_actual"], dtype=np.float32)), "initial_measured_joints")

    starts = np.arange(1, n, 4)
    raw, sent = physics["q_target_raw"], physics["q_target"]
    prior = physics["q_actual"][0, :6].astype(float)
    ema_error, hold_error = 0., 0.
    for index in starts:
        expected = .8*raw[index, :6]+.2*prior
        ema_error = max(ema_error, max_error(sent[index, :6], expected))
        hold_error = max(hold_error, max_error(sent[index:min(index+4, n)], sent[index]))
        hold_error = max(hold_error, max_error(raw[index:min(index+4, n)], raw[index]))
        prior = sent[index, :6]
    measures.update(ema_max_error_rad=ema_error, held_target_max_error=hold_error, executed_actions=len(starts))
    check(ema_error <= 1e-7, "ema_recurrence_from_measured_reset_state")
    check(hold_error == 0., "raw_and_sent_targets_held_four_physics_ticks")
    check(np.array_equal(raw[:, 6:], sent[:, 6:]), "finger_commands_unfiltered_exactly")
    check(len(starts) == policy["commands_selected"] == policy["smoothing"]["applied_actions"], "command_counts")
    expected_request = np.r_[-1, np.repeat(np.arange(len(starts))//32, 4)[:n-1]]
    expected_action = np.r_[-1, np.repeat(np.arange(len(starts))%32, 4)[:n-1]]
    check(np.array_equal(expected_request, physics["policy_request"]) and
          np.array_equal(expected_action, physics["policy_action_index"]), "sequential_chunk_and_action_indices")

    requests = [json.loads(line) for line in (directory/"policy_requests.jsonl").read_text().splitlines() if line.strip()]
    check(len(requests) == policy["inference_requests"], "inference_request_count")
    check([request["request_index"] for request in requests] == list(range(len(requests))), "sequential_request_ids")
    expected_requests = (len(starts)+31)//32
    invalid_action = report.get("failure_reason") == "invalid_policy_action"
    check(len(requests) == expected_requests+int(invalid_action), "request_count_including_rejected_last_chunk")
    model_raw_error = state_error = 0.
    for request in requests:
        index = request["request_index"]
        sample = index*32*4
        check(request["simulation_time_s"] == sample*dt, "request_time_at_chunk_boundary", request=index)
        if sample < n:
            q = physics["q_actual"][sample].astype(float)
            expected_state = np.r_[np.rad2deg(q[:6]), 100*(q[6]-q[7])].astype(np.float32)
            state_error = max(state_error, max_error(request["state_raw"], expected_state))
        if request["status"] == "failed":
            check(invalid_action and index == len(requests)-1, "only_invalid_final_request_rejected")
            continue
        binary_response = unpack((directory/request["wire_prefix"]).with_suffix(".response.msgpack").read_bytes())
        action = np.asarray(binary_response["action"], dtype=float).reshape(32, 7)
        converted = np.column_stack((np.deg2rad(action[:, :6]), np.clip(action[:, 6]*.01, 0., .1)))
        full = np.column_stack((converted[:, :6], converted[:, 6]/2, -converted[:, 6]/2))
        selected_starts = starts[physics["policy_request"][starts] == index]
        if len(selected_starts):
            model_raw_error = max(model_raw_error, max_error(raw[selected_starts], full[:len(selected_starts)]))
    measures.update(model_to_raw_target_max_error=model_raw_error, observation_state_max_error=state_error)
    check(model_raw_error <= 1e-12, "raw_targets_match_exact_binary_model_responses")
    check(state_error == 0., "inference_observations_match_measured_physics_at_chunk_boundary")

    contacts = read(directory/"contacts.json")
    counts = Counter(contact["finger"] for contact in contacts)
    tick_contacts = {int(contact["tick"]) for contact in contacts}
    check(all(0 <= contact["tick"] < n for contact in contacts), "contact_tick_bounds")
    check({finger: counts[finger] for finger in ("link7", "link8")} == report["contact_counts"], "contact_event_counts")
    both = all(counts[finger] > 0 for finger in ("link7", "link8"))
    check(both == report["gripper_contact_both_fingers"], "bilateral_contact_history")
    cavity = scene["box_inside"]
    box = dict(center_world=cavity["origin_world"], yaw_deg=cavity["yaw_deg"],
               inner_size_xy=(np.array(cavity["max_local"])-np.array(cavity["min_local"]))[:2],
               floor_z_world=cavity["origin_world"][2]+cavity["min_local"][2],
               rim_z_world=cavity["origin_world"][2]+cavity["max_local"][2])
    outcome = PickPlaceOutcome(scene["object_geometry"], config["table_height"], box)
    for index, seconds in enumerate(physics["time_s"]):
        q = physics["q_actual"][index]
        recomputed = outcome.update(seconds, physics["cube_position"][index], physics["cube_quaternion"][index],
                                    physics["cube_velocity"][index], float(q[6]-q[7]), finger_contact=index in tick_contacts)
    for key, value in recomputed.items():
        saved = report["physical_outcome"][key]
        check(value == saved if isinstance(value, (bool, np.bool_)) or value is None else abs(value-saved) <= 1e-10,
              "recomputed_physical_outcome_"+key, recomputed=value.item() if isinstance(value, np.generic) else value,
              reported=saved)
    recomputed_success = bool(recomputed["success"] and both)
    check(recomputed_success == report["physical_success"], "physical_success_recomputed_with_contact_history")
    if invalid_action:
        check(not report["physical_success"] and report["evaluation_status"] == "complete", "invalid_action_is_completed_failure")
    elif not report["physical_success"]:
        check(report["failure_reason"] == "timeout" and abs(report["duration_s"]-20.) < 1e-12, "timeout_failure_horizon")
    else:
        check(report["failure_reason"] is None and report["termination"] == "success", "success_termination")
    return dict(episode_id=episode["episode_id"], group=episode["group"], color=episode["color"],
                physical_success=report["physical_success"], failure_reason=report["failure_reason"],
                passed=not issues, issues=issues, measures=measures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Evaluation directory containing plan.json and run/")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    plan = read(args.root/"plan.json")
    config = read(args.root/"effective_config.json")
    episodes = plan["episodes"]
    ids = [episode["episode_id"] for episode in episodes]
    groups = {name: [episode for episode in episodes if episode["group"] == name]
              for name in ("position_red", "position_color")}
    plan_checks = dict(unique_ids=len(ids) == len(set(ids)), two_groups_100_each=all(len(v) == 100 for v in groups.values()),
                       red_group_fixed_red=all(e["color"] == "red" for e in groups["position_red"]),
                       colors_balanced=Counter(e["color"] for e in groups["position_color"]) ==
                                       {color: 20 for color in ("red", "green", "blue", "white", "yellow")},
                       positions_paired=[e["center_xy"] for e in groups["position_red"]] ==
                                        [e["center_xy"] for e in groups["position_color"]])
    halfplanes = np.array(plan["region"]["marker_halfplanes"])
    corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])*.015
    plan_checks["all_cube_footprints_inside_markers"] = all(
        np.max((np.array(e["center_xy"])+corners) @ halfplanes[:, :2].T+halfplanes[:, 2]) <= 1e-10 for e in episodes)
    completed, pending, infrastructure = [], [], []
    for episode in episodes:
        path = args.root/"run"/"episodes"/episode["episode_id"]/"report.json"
        try:
            report = read(path)
        except (FileNotFoundError, json.JSONDecodeError):
            pending.append(episode["episode_id"])
            continue
        if report.get("evaluation_status") == "complete":
            completed.append(episode)
        else:
            infrastructure.append(dict(episode_id=episode["episode_id"], evaluation_status=report.get("evaluation_status"),
                                       error=report.get("evaluation_error")))
    selected = completed if args.limit is None else completed[:args.limit]
    baseline = read(args.root/"run"/"episodes"/completed[0]["episode_id"]/"report.json")["policy"] if completed else None
    audits = []
    for episode in selected:
        try:
            audit = audit_episode(args.root/"run"/"episodes"/episode["episode_id"], episode, config, baseline)
        except Exception as error:
            audit = dict(episode_id=episode["episode_id"], passed=False,
                         issues=[dict(check="audit_execution", error=f"{type(error).__name__}: {error}")])
        audits.append(audit)
    passed = all(plan_checks.values()) and all(audit["passed"] for audit in audits)
    complete = len(completed) == len(episodes) and len(selected) == len(episodes) and not infrastructure
    numeric_keys = sorted({key for audit in audits for key in audit.get("measures", {})})
    summary = dict(version=1, full_audit_complete=complete, checked_passed=passed, planned=len(episodes),
                   completed_reports=len(completed), audited=len(audits), pending=pending,
                   infrastructure_errors=infrastructure, plan_checks=plan_checks,
                   endpoint=None if baseline is None else baseline["endpoint"],
                   server_metadata=None if baseline is None else baseline["server_metadata"],
                   checked_group_counts=dict(Counter(a["group"] for a in audits if "group" in a)),
                   checked_failures_by_reason=dict(Counter(a["failure_reason"] for a in audits if a.get("failure_reason"))),
                   aggregate_maxima={key: max(a["measures"][key] for a in audits if key in a.get("measures", {}))
                                     for key in numeric_keys},
                   audit_failures=[audit for audit in audits if not audit["passed"]], episodes=audits,
                   scope="Recorded commands, exact binary policy responses, physics, contact events and authored scene metadata. Material colors are checked against authored metadata, not estimated from rendered pixels. Partial audits do not establish full evaluation completion.")
    path = args.root/("rollout_audit.json" if args.limit is None else "rollout_audit_sample.json")
    save(path, summary)
    print(json.dumps(dict(full_audit_complete=complete, checked_passed=passed, completed_reports=len(completed),
                          audited=len(audits), audit_failures=len(summary["audit_failures"]), path=str(path))))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
