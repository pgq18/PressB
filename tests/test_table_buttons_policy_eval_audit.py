"""Independent closed-loop outcome, temporal alignment and action provenance tests."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("tb_policy_audit", ROOT/"scripts/audit_table_buttons_policy_eval.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def physical_records():
    q = np.zeros((9, 8))
    q[:, 6], q[:, 7] = .0001, -.0001
    physics = dict(time_s=np.arange(9)/120, q_actual=q.copy(), q_target=q.copy(),
        travel_m=np.zeros((9, 3)), finger_force_n=np.zeros((9, 3)))
    physics["travel_m"][4:, 0] = .0032
    physics["finger_force_n"][4:, 0] = 1.
    return physics


def contacts_for(physics):
    return [dict(tick=tick, color=color, fingers=["link7"], force_n=float(physics["finger_force_n"][tick, index]))
            for tick in range(len(physics["time_s"])) for index, color in enumerate(audit.COLORS)
            if physics["finger_force_n"][tick, index] > 0]


def test_four_samples_are_needed_to_span_25ms_not_three():
    p = physical_records()
    good = audit.physical_outcome(p, {"success": audit.THRESHOLDS}, "red", contacts_for(p), [])
    assert good["success"] and good["first_success_time_s"] == pytest.approx(7/120)
    p["finger_force_n"][7:, 0] = 0
    bad = audit.physical_outcome(p, {"success": audit.THRESHOLDS}, "red", contacts_for(p), [])
    assert not bad["success"] and not bad["target_pressed"]


def test_dynamic_float32_fingers_are_promoted_before_full_width_subtraction():
    p = physical_records()
    p["q_actual"] = p["q_actual"].astype(np.float32)
    # Unequal finger coordinates are normal dynamic tracking observations.
    p["q_actual"][4:, 6] = np.float32(.047123)
    p["q_actual"][4:, 7] = np.float32(-.043456)
    q = p["q_actual"][::4].astype(np.float64)
    states = np.column_stack((q[:, :6], q[:, 6]-q[:, 7]))
    rounded_first = p["q_actual"][::4, 6]-p["q_actual"][::4, 7]
    assert np.max(np.abs(states[:, 6]-rounded_first)) > 1e-10
    action = np.zeros((3, 7)); action[:, 6] = .0002
    frames = dict(sim_time=np.arange(3)/30, state=states, action=action)
    assert audit.audit_timing(frames, p) == (3, 9)
    # Keep the original strict threshold: a real state corruption still fails.
    frames["state"][1, 6] += 2e-10
    with pytest.raises(ValueError, match="State/RGB boundary"):
        audit.audit_timing(frames, p)


def test_zero_action_float32_finger_state_uses_same_promotion_order():
    p = {key: value[:1].copy() for key, value in physical_records().items()}
    p["q_actual"] = p["q_actual"].astype(np.float32)
    p["q_actual"][0, 6:] = [.047123, -.043456]
    q = p["q_actual"][0].astype(np.float64)
    frames = dict(sim_time=np.array([0.]), state=np.r_[q[:6], q[6]-q[7]][None], action=np.zeros((1, 7)))
    assert audit.audit_timing(frames, p, invalid_policy=True) == (1, 1)


@pytest.mark.parametrize("kind", ["force_interruption", "wrong_contact_after_target", "wrong_press_after_target", "collision_after_target"])
def test_physical_failures_cannot_hide_behind_latched_target(kind):
    p = physical_records()
    unexpected = []
    if kind == "force_interruption":
        p["finger_force_n"][6, 0] = 0
    elif kind == "wrong_contact_after_target":
        p["finger_force_n"][8, 1] = .021
    elif kind == "wrong_press_after_target":
        p["travel_m"][5:, 1] = .0031
        p["finger_force_n"][5:, 1] = .1
    else:
        unexpected = [dict(tick=8, actors=["/World/Piper/link2", "/World/Table/Top"], force_n=.021)]
    result = audit.physical_outcome(p, {"success": audit.THRESHOLDS}, "red", contacts_for(p), unexpected)
    assert not result["success"]
    if "after_target" in kind:
        assert result["target_pressed"]
    if kind == "wrong_press_after_target":
        assert result["wrong_press"] and result["press_events"][-1]["color"] == "yellow"


def test_contact_matrix_must_have_event_evidence():
    p = physical_records()
    with pytest.raises(ValueError, match="Contact log"):
        audit.physical_outcome(p, {"success": audit.THRESHOLDS}, "red", contacts_for(p)[:-1], [])


def control_records():
    lower, upper = np.array([-1.]*6+[0.]), np.array([1.]*6+[.1])
    actions = np.zeros((7, 7))
    actions[:, 6] = .02
    actions[0, 0] = 180.
    actions[1, 6] = 20.
    applied = np.clip(actions[:2]*audit.TO_SIM, lower, upper)
    states = np.zeros((3, 7)); states[:, 6] = .0002
    frames = dict(sim_time=np.arange(3)/30, state=states, action=np.vstack([applied, applied[-1]]))
    model = dict(model_id="near_red", model_sha256="a"*64)
    cell = dict(model_id="near_red", instruction="Press the red button.")
    row = dict(policy_seed=2000000)
    requests = [dict(frame_index=0, sim_time_s=0., request_id=0, noise_seed=2000000,
        state_deg_cm=(states[0]/audit.TO_SIM).tolist(), instruction=cell["instruction"],
        model=model, raw_actions_deg_cm=actions.tolist(), normalized_actions=(actions/audit.SCALE).tolist(), round_trip_seconds=.1)]
    controls = []
    for index in range(2):
        converted = actions[index]*audit.TO_SIM
        controls.append(dict(frame_index=index, request_id=0, action_index=index,
            raw_command_deg_cm=actions[index].tolist(), converted_rad_m=converted.tolist(),
            clipped_rad_m=applied[index].tolist(), applied_rad_m=applied[index].tolist(),
            clipped_dimensions=np.flatnonzero(converted != applied[index]).tolist()))
    return frames, requests, controls, lower, upper, cell, row, model


def test_all_model_commands_verified_including_full_gripper_clipping():
    frames, req, controls, low, high, cell, row, model = control_records()
    result = audit.audit_controls(frames, req, controls, low, high, cell=cell, row=row, model=model)
    assert result["clipped_actions"] == 2
    assert result["clipped_by_dimension"] == [1, 0, 0, 0, 0, 0, 1]
    assert result["maximum_clamp_deg_cm"][6] == pytest.approx(10.)
    assert frames["action"][1, 6] == .1  # model opens gripper; never forced to training .0002m


@pytest.mark.parametrize("kind,match", [
    ("wrong_language", "language"), ("wrong_noise_seed", "RNG"), ("half_gripper", "state units"),
    ("wrong_chunk_index", "corresponding model"), ("expert_substitution", "substitution"),
    ("cm_as_m", "conversion"), ("wrong_normalization", "normalization"),
    ("hidden_clamp", "dimension count"), ("model_changed", "identity"),
])
def test_action_provenance_corruptions_rejected(kind, match):
    frames, req, controls, low, high, cell, row, model = control_records()
    if kind == "wrong_language": req[0]["instruction"] = "Press the blue button."
    elif kind == "wrong_noise_seed": req[0]["noise_seed"] += 1
    elif kind == "half_gripper": req[0]["state_deg_cm"][6] /= 2
    elif kind == "wrong_chunk_index": controls[0]["action_index"] = 1
    elif kind == "expert_substitution": controls[0]["applied_rad_m"][0] = .5
    elif kind == "cm_as_m": controls[1]["converted_rad_m"][6] = 20.
    elif kind == "wrong_normalization": req[0]["normalized_actions"][0][6] *= 2
    elif kind == "hidden_clamp": controls[0]["clipped_dimensions"] = []
    else: req[0]["model"] = dict(model, model_sha256="b"*64)
    with pytest.raises(ValueError, match=match):
        audit.audit_controls(frames, req, controls, low, high, cell=cell, row=row, model=model)


def test_zero_action_invalid_model_retains_failure_record():
    frames, req, controls, low, high, cell, row, model = control_records()
    frames = {name: array[:1].copy() for name, array in frames.items()}
    req[0].pop("raw_actions_deg_cm"); req[0].pop("normalized_actions")
    req[0]["error"] = "InvalidPolicyAction"
    result = audit.audit_controls(frames, req, [], low, high, cell=cell, row=row, model=model, invalid_policy=True)
    assert result["policy_requests"] == 1 and result["total_actions"] == 0


def test_failing_last_request_does_not_change_successful_chunk_history():
    frames, req, controls, low, high, cell, row, model = control_records()
    req.append(dict(req[0], frame_index=7, request_id=1, noise_seed=2000001, error="invalid"))
    with pytest.raises(ValueError, match="boundary"):
        audit.audit_controls(frames, req, controls, low, high, cell=cell, row=row, model=model, invalid_policy=True)


def test_wilson_and_denominator_do_not_hide_invalid_completed_episode():
    record = dict(success=True, stop_reason="success", first_success_time_s=3.,
        wrong_press=False, non_target_contact=False, unexpected_collision=False, invalid_policy_action=False,
        commands=dict(total_actions=90, clipped_actions=1, inference_round_trip_seconds=[.1, .2]))
    result = audit.summary([record], 3, [dict(seed=10002)], [dict(seed=10001)])
    assert result["success_rate"] is None and result["success_rate_over_completed"] == .5
    assert result["success_rate_planned_lower_bound"] == pytest.approx(1/3)
    assert (result["completed"], result["audited"], result["invalid"], result["missing"]) == (2, 1, 1, 1)
    assert audit.wilson(100, 100)[0] == pytest.approx(.9630065018)
    assert audit.wilson(0, 100)[1] == pytest.approx(.0369934982)


def write_complete_episode(tmp_path):
    directory = tmp_path/"episode_000000"
    directory.mkdir()
    cfg = json.loads((ROOT/"configs/piper_table_buttons.json").read_text())
    cfg["position_sampling"]["spacing_profile"] = "near"
    cfg["task_color"], cfg["task_instruction"] = "red", "Press the red button."
    target = np.r_[cfg["initial_joint_positions_rad"], cfg["initial_gripper_width_m"]]
    limits = np.vstack([audit.arm_limits(str(ROOT/cfg["robot_urdf"])), [0., .05], [-.05, 0.]])
    lower, upper = np.r_[limits[:6, 0], 0.], np.r_[limits[:6, 1], .1]
    action = np.tile(target, (3, 1))
    physical = physical_records()
    joints = np.r_[target[:6], target[6]/2, -target[6]/2]
    physical["q_actual"][:] = joints
    physical["q_target"][:] = joints
    frames = dict(sim_time=np.arange(3)/30, state=action.copy(), action=action.copy())
    model = dict(model_id="near_red", model_sha256="a"*64, strict_loaded=True, model_digest_verified=True,
        camera_order=["up", "wrist"], image_input_size=[640, 480], image_model_size=[224, 224],
        normalization_scale=audit.SCALE.tolist(), action_horizon=7, action_dim=7)
    cell = dict(cell_id="near_red", model_id="near_red", spacing_profile="near", target_color="red", instruction=cfg["task_instruction"])
    row = dict(episode_index=0, seed=10000, policy_seed=2000000,
        effective_config_file=str(directory/"config.json"), layout_sha256="b"*64, effective_config_sha256="c"*64)
    commands = dict(lower_rad_m=lower.tolist(), upper_rad_m=upper.tolist(), smoothing=dict(mode="none"),
        total_actions=2, clipped_actions=0, clipped_by_dimension=[0]*7)
    common = dict(**cell, seed=10000, success=True, stop_reason="success", duration_s=2/30, model=model, expert_actions_used=False)
    metadata = dict(common, schema_version="pressb.table_buttons.policy_eval.v1", fps=30, physics_dt=1/120,
        state_units=["rad"]*6+["m"], action_units=["rad"]*6+["m"], gripper_representation="full_width",
        policy_normalization_scale=audit.SCALE.tolist(), initial_state_measured_rad_m=target.tolist(),
        model_sha256=model["model_sha256"], config_sha256=audit.canonical_sha(cfg), layout_sha256=row["layout_sha256"])
    report = dict(common, evaluation_status="complete", policy_or_RL_evaluation=True,
        network_wait_advances_simulation=False, config=cfg, joint_limits_dof_rad_m=limits.tolist(), commands=commands)
    raw = np.tile(target/audit.TO_SIM, (7, 1))
    requests = [dict(frame_index=0, sim_time_s=0., request_id=0, noise_seed=row["policy_seed"],
        state_deg_cm=raw[0].tolist(), instruction=cell["instruction"], model=model,
        raw_actions_deg_cm=raw.tolist(), normalized_actions=(raw/audit.SCALE).tolist(), round_trip_seconds=.1)]
    controls = [dict(frame_index=i, request_id=0, action_index=i, raw_command_deg_cm=raw[i].tolist(),
        converted_rad_m=target.tolist(), clipped_rad_m=target.tolist(), applied_rad_m=target.tolist(), clipped_dimensions=[])
        for i in range(2)]
    for name, value in (("metadata", metadata), ("report", report), ("config", cfg),
                        ("contacts", contacts_for(physical)), ("unexpected_contacts", [])):
        (directory/f"{name}.json").write_text(json.dumps(value))
    for name, rows in (("policy_requests", requests), ("control", controls)):
        (directory/f"{name}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    np.savez_compressed(directory/"data.npz", **frames)
    np.savez_compressed(directory/"physics.npz", **physical)
    return directory, cell, row, dict(defaults=dict(timeout_s=12.))


def test_complete_episode_audits_all_original_evidence(tmp_path):
    args = write_complete_episode(tmp_path)
    result = audit.audit_episode(*args)
    assert result["audit_valid"] and result["success"]
    assert result["physics_samples"] == 9 and result["commands"]["total_actions"] == 2


def test_reported_success_cannot_override_raw_failure(tmp_path):
    directory, cell, row, plan = write_complete_episode(tmp_path)
    physics = audit.load_arrays(directory)[1]
    physics["finger_force_n"][8, 1] = .2
    np.savez_compressed(directory/"physics.npz", **physics)
    (directory/"contacts.json").write_text(json.dumps(contacts_for(physics)))
    with pytest.raises(ValueError, match="success disagrees"):
        audit.audit_episode(directory, cell, row, plan)


def test_paired_comparison_uses_same_layout_only():
    rows = [dict(cell_id="near_red", seed=10000, success=True),
            dict(cell_id="near_multitask_red", seed=10000, success=False),
            dict(cell_id="near_multitask_red", seed=10001, success=True)]
    result = audit.paired_comparisons(rows)["near_red"]
    assert result["paired_audited_episodes"] == 1
    assert result["single_only_success"] == 1
    assert result["multitask_minus_single_success_rate"] == -1.


def test_exact_finger_closure_has_distinct_task_and_collector_outcomes():
    p = physical_records()
    closure = [dict(tick=2, actors=["/World/Piper/link8", "/World/Piper/link7"], force_n=.2)]
    old = audit.physical_outcome(p, {"success": audit.THRESHOLDS}, "red", contacts_for(p), closure)
    new = audit.physical_outcome(p, {"success": audit.THRESHOLDS}, "red", contacts_for(p), closure, audit.DUAL_CONTACT_POLICY)
    assert not old["success"] and old["first_terminal_tick"] == 2
    assert new["task_success"] and new["success"] and not new["strict_collector_success"]
    assert new["first_terminal_tick"] == 7 and new["strict_first_terminal_tick"] == 2
    assert new["first_strict_failure_time_s"] == pytest.approx(2/120)
    assert new["finger_closure_contact_count"] == 1 and new["blocking_contact_count"] == 0


@pytest.mark.parametrize("other", ["/World/Piper/link6", "/World/Table/Top", "/World/TableButtons/blue/Cap"])
def test_no_other_contact_is_exempt_under_dual_policy(other):
    p = physical_records()
    collision = [dict(tick=8, actors=["/World/Piper/link7", other], force_n=.03)]
    result = audit.physical_outcome(p, {"success": audit.THRESHOLDS}, "red", contacts_for(p), collision, audit.DUAL_CONTACT_POLICY)
    assert not result["success"] and not result["strict_collector_success"]
    assert result["blocking_contact_count"] == 1 and result["finger_closure_contact_count"] == 0


def test_cannot_silently_add_forgiving_contact_exceptions():
    p = physical_records()
    changed = deepcopy(audit.DUAL_CONTACT_POLICY)
    changed["allowed_task_contact_pairs"].append(["/World/Piper/link6", "/World/Piper/link7"])
    with pytest.raises(ValueError, match="Unrecognized collision policy"):
        audit.physical_outcome(p, {"success": audit.THRESHOLDS}, "red", contacts_for(p), [], changed)


def test_complete_dual_episode_keeps_both_original_and_classified_contacts(tmp_path):
    directory, cell, row, plan = write_complete_episode(tmp_path)
    closure = [dict(tick=2, actors=["/World/Piper/link7", "/World/Piper/link8"], force_n=.2)]
    plan["defaults"]["collision_policy"] = audit.DUAL_CONTACT_POLICY
    for name in ("report", "metadata"):
        data = audit.read_json(directory/f"{name}.json")
        data.update(task_success=True, strict_collector_success=False, collision_policy=audit.DUAL_CONTACT_POLICY,
                    finger_closure_contact_count=1, blocking_contact_count=0)
        (directory/f"{name}.json").write_text(json.dumps(data))
    for name, value in (("unexpected_contacts", closure), ("finger_closure_contacts", closure), ("blocking_contacts", [])):
        (directory/f"{name}.json").write_text(json.dumps(value))
    result = audit.audit_episode(directory, cell, row, plan)
    assert result["success"] and not result["strict_collector_success"]
    (directory/"unexpected_contacts.json").write_text("[]")
    with pytest.raises(ValueError, match="closure split"):
        audit.audit_episode(directory, cell, row, plan)
