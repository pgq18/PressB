"""Native joint6 evidence and a fixed-gripper wire must remain distinguishable."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("joint7_audit_fixtures", ROOT / "tests/test_table_buttons_policy_eval_audit.py")
fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixtures)
audit = fixtures.audit


MODEL_IDS = tuple(audit.JOINT6_MODELS)
EXPANDED_MODEL_IDS = tuple(model_id for model_id in MODEL_IDS if model_id.endswith("_1000"))


def identity(model_id="medium_red_joint6"):
    model = dict(model_id=model_id, model_sha256="a" * 64, best_step=492,
        strict_loaded=True, model_digest_verified=True, camera_order=["up", "wrist"],
        image_input_size=[640, 480], image_model_size=[224, 224], normalization_scale=audit.SCALE.tolist(),
        native_schema=audit.JOINT6_SCHEMA, native_action_dim=6, state_dim=7,
        action_dim=7, wire_action_dim=7, wire_schema="pressb.table_buttons.joint7_policy.v1", action_horizon=7,
        noise_shape_per_sample=[1, 7, 6], noise_hash_array_shape=[7, 6], noise_hash_dtype="float32",
        noise_response_field="initial_noise", noise_hash_memory_order="C",
        native_action_units=["degree"] * 6, native_action_normalization_scale=[180.] * 6,
        fixed_gripper_full_width_m=audit.FIXED_GRIPPER_FULL_WIDTH_M,
        fixed_gripper_full_width_cm=audit.FIXED_GRIPPER_FULL_WIDTH_M * 100., gripper_learned=False)
    if model_id in EXPANDED_MODEL_IDS:
        profile, source_count = audit.JOINT6_MODELS[model_id]
        model.update(source_dataset_group=f"{profile}_red", source_episode_count=source_count)
    return model


def contract(model):
    keys = audit.JOINT6_CONTRACT_KEYS + (("source_dataset_group", "source_episode_count")
        if model["model_id"] in EXPANDED_MODEL_IDS else ())
    return {key: model[key] for key in keys}


def add_native_evidence(request):
    request["native_actions"] = np.asarray(request["raw_actions_deg_cm"])[:, :6].tolist()
    request["native_normalized_actions"] = np.asarray(request["normalized_actions"])[:, :6].tolist()
    request["normalized_state"] = (np.asarray(request["state_deg_cm"]) / audit.SCALE).tolist()
    noise = np.arange(42, dtype=np.float32).reshape(7, 6) / np.float32(42)
    request["initial_noise"] = noise.tolist()
    request["initial_noise_sha256"] = hashlib.sha256(noise.tobytes()).hexdigest()


def joint6_records(model_id="medium_red_joint6"):
    frames, requests, controls, lower, upper, cell, row, _ = fixtures.control_records()
    model = identity(model_id)
    cell.update(model_id=model["model_id"], model_contract=contract(model))
    if model_id in EXPANDED_MODEL_IDS:
        cell.update(spacing_profile=audit.JOINT6_MODELS[model_id][0], target_color="red")
    request = requests[0]
    request["model"] = deepcopy(model)
    raw = np.asarray(request["raw_actions_deg_cm"])
    raw[:, 6] = audit.FIXED_GRIPPER_FULL_WIDTH_M * 100.
    request["raw_actions_deg_cm"] = raw.tolist()
    request["normalized_actions"] = (raw / audit.SCALE).tolist()
    # Measured input width differs from fixed command; it must not be replaced.
    frames["state"][:, 6] = .00021
    request["state_deg_cm"] = (frames["state"][0] / audit.TO_SIM).tolist()
    add_native_evidence(request)
    for index, control in enumerate(controls):
        converted = raw[index] * audit.TO_SIM
        clipped = np.clip(converted, lower, upper)
        control.update(raw_command_deg_cm=raw[index].tolist(), converted_rad_m=converted.tolist(),
            clipped_rad_m=clipped.tolist(), applied_rad_m=clipped.tolist(),
            clipped_dimensions=np.flatnonzero(converted != clipped).tolist())
        frames["action"][index] = clipped
    frames["action"][-1] = frames["action"][-2]
    return frames, requests, controls, lower, upper, cell, row, model


def evaluate(records):
    frames, requests, controls, lower, upper, cell, row, model = records
    return audit.audit_controls(frames, requests, controls, lower, upper, cell=cell, row=row, model=model)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_joint6_native_noise_and_fixed_gripper_with_measured_state7(model_id):
    data = joint6_records(model_id)
    result = evaluate(data)
    assert result["native_action_audit"]["verified_native_noise_samples"] == 1
    assert result["native_action_audit"]["noise_shape_per_sample"] == [1, 7, 6]
    assert result["clipped_by_dimension"] == [1, 0, 0, 0, 0, 0, 0]
    assert data[0]["state"][0, 6] != data[0]["action"][0, 6]


@pytest.mark.parametrize("model_id", MODEL_IDS)
@pytest.mark.parametrize("corruption,match", [
    ("native_changed", "constructing"), ("native_seven", "native joint6 action"),
    ("noise_seven", "initial_noise"), ("noise_tampered", "SHA256"),
    ("missing_noise", "initial_noise"), ("noise_nonfinite", "initial_noise"),
    ("wrong_gripper", "fixed dataset"), ("metadata_seven_noise", "noise_shape_per_sample"),
    ("wrong_frozen_sha", "frozen joint6 contract"), ("missing_normalized_state", "state evidence"),
    ("fixed_observed_gripper", "normalized measured state"),
])
def test_joint6_corrupted_evidence_rejected(model_id, corruption, match):
    data = joint6_records(model_id)
    request, model, cell = data[1][0], data[-1], data[-3]
    if corruption == "native_changed": request["native_actions"][0][0] += 1.
    elif corruption == "native_seven": request["native_actions"] = request["raw_actions_deg_cm"]
    elif corruption == "noise_seven": request["initial_noise"] = np.zeros((7, 7)).tolist()
    elif corruption == "noise_tampered": request["initial_noise"][0][0] += 1.
    elif corruption == "missing_noise": request.pop("initial_noise")
    elif corruption == "noise_nonfinite": request["initial_noise"][0][0] = float("nan")
    elif corruption == "wrong_gripper":
        request["raw_actions_deg_cm"][0][6] = .03
        request["normalized_actions"][0][6] = .03 / 8
    elif corruption == "metadata_seven_noise": model["noise_shape_per_sample"] = [1, 7, 7]
    elif corruption == "wrong_frozen_sha": cell["model_contract"]["model_sha256"] = "b" * 64
    elif corruption == "missing_normalized_state": request.pop("normalized_state")
    else: request["normalized_state"][6] = .02 / 8
    with pytest.raises(ValueError, match=match): evaluate(data)


def test_joint6_complete_episode_retains_physical_and_action_audits(tmp_path):
    directory, cell, row, plan = fixtures.write_complete_episode(tmp_path)
    model = identity()
    cell.update(cell_id=model["model_id"], model_id=model["model_id"], model_contract=contract(model))
    for name in ("metadata", "report"):
        value = audit.read_json(directory / f"{name}.json")
        value.update(cell_id=cell["cell_id"], model_id=cell["model_id"], model=deepcopy(model))
        (directory / f"{name}.json").write_text(json.dumps(value))
    requests = audit.read_lines(directory / "policy_requests.jsonl")
    requests[0]["model"] = deepcopy(model)
    add_native_evidence(requests[0])
    (directory / "policy_requests.jsonl").write_text(json.dumps(requests[0]) + "\n")
    result = audit.audit_episode(directory, cell, row, plan)
    assert result["success"] and result["strict_collector_success"]
    assert result["commands"]["native_action_audit"]["native_action_dim"] == 6


def scoped_plan(tmp_path, model_id="medium_red_joint6"):
    profile = audit.JOINT6_MODELS[model_id][0]
    dataset_group = f"{profile}_red"
    rows = [dict(episode_index=i, seed=10000+i, policy_seed=3000000+i*1000) for i in range(100)]
    original = dict(cell_id=dataset_group, episodes=rows)
    reference = dict(schema="pressb.table_buttons.policy_eval_plan.v1", config={}, source={},
        defaults=dict(collision_policy=audit.DUAL_CONTACT_POLICY), cells=[original])
    path = tmp_path / "original.json"
    path.write_text(json.dumps(reference))
    plan = deepcopy(reference)
    plan.update(planned_episodes=100, evaluation_scope=dict(kind="joint6_single_model", model_id=model_id,
        reference_plan_file=str(path), reference_plan_sha256=audit.file_sha(str(path)), reference_cell_id=dataset_group))
    plan["cells"] = [dict(cell_id=model_id, model_id=model_id, model_type="single",
        spacing_profile=profile, target_color="red", instruction="Press the red button.",
        episodes=rows, model_contract=contract(identity(model_id)))]
    return plan


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_joint6_scope_uses_all_exact_reference_rows(tmp_path, model_id):
    plan = scoped_plan(tmp_path, model_id)
    assert audit.audit_joint6_plan_scope(plan)["reference_cell_id"] == f"{audit.JOINT6_MODELS[model_id][0]}_red"


@pytest.mark.parametrize("model_id", MODEL_IDS)
@pytest.mark.parametrize("corruption,match", [
    ("changed_seed", "100 original"), ("missing_row", "100 original"),
    ("changed_protocol", "defaults"), ("extra_cell", "exactly one"),
    ("wrong_reference_sha", "reference plan changed"), ("wrong_contract", "native_action_dim"),
])
def test_joint6_scope_cannot_silently_change_comparison(tmp_path, model_id, corruption, match):
    plan = scoped_plan(tmp_path, model_id)
    if corruption == "changed_seed": plan["cells"][0]["episodes"][0]["seed"] = 11000
    elif corruption == "missing_row": plan["cells"][0]["episodes"].pop()
    elif corruption == "changed_protocol": plan["defaults"]["smoothing"] = dict(mode="ema")
    elif corruption == "extra_cell": plan["cells"].append(deepcopy(plan["cells"][0]))
    elif corruption == "wrong_reference_sha": plan["evaluation_scope"]["reference_plan_sha256"] = "0" * 64
    else: plan["cells"][0]["model_contract"]["native_action_dim"] = 7
    with pytest.raises(ValueError, match=match): audit.audit_joint6_plan_scope(plan)


@pytest.mark.parametrize("model_id", EXPANDED_MODEL_IDS)
@pytest.mark.parametrize("field,value", [
    ("source_dataset_group", "near_multitask"), ("source_episode_count", 100),
    ("source_dataset_group", None), ("source_episode_count", None),
])
def test_expanded_joint6_requires_matching_training_data(tmp_path, model_id, field, value):
    plan = scoped_plan(tmp_path, model_id)
    plan["cells"][0]["model_contract"][field] = value
    with pytest.raises(ValueError, match=field):
        audit.audit_joint6_plan_scope(plan)
    data = joint6_records(model_id)
    data[-1][field] = value
    with pytest.raises(ValueError, match=field):
        evaluate(data)


@pytest.mark.parametrize("model_id", EXPANDED_MODEL_IDS)
def test_expanded_joint6_cannot_use_another_spacing(tmp_path, model_id):
    plan = scoped_plan(tmp_path, model_id)
    profile = audit.JOINT6_MODELS[model_id][0]
    other = next(value for value in audit.PROFILES if value != profile)
    plan["evaluation_scope"]["reference_cell_id"] = f"{other}_red"
    with pytest.raises(ValueError, match=f"original {profile}_red layouts"):
        audit.audit_joint6_plan_scope(plan)
    data = joint6_records(model_id)
    data[-3]["spacing_profile"] = other
    with pytest.raises(ValueError, match="selected task"):
        evaluate(data)
