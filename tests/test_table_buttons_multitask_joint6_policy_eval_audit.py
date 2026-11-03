"""Multi-task evaluation preserves color pairing and one six-joint checkpoint."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("joint6_audit_fixtures", ROOT / "tests/test_table_buttons_joint6_policy_eval_audit.py")
fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixtures)
audit = fixtures.audit


def identity(profile):
    model = fixtures.identity()
    model.update(model_id=f"{profile}_multitask_joint6_3000", **audit.multitask_model_contract(profile))
    return model


def contract(profile):
    model = identity(profile)
    return {key: model[key] for key in audit.JOINT6_MULTITASK_CONTRACT_KEYS}


def scoped_plan(tmp_path, profile="near"):
    originals = []
    for index, color in enumerate(audit.COLORS):
        rows = [dict(episode_index=i, seed=10000+i, policy_seed=2000000+index*100000+i*1000)
                for i in range(100)]
        originals.append(dict(cell_id=f"{profile}_multitask_{color}", episodes=rows))
    reference = dict(schema="pressb.table_buttons.policy_eval_plan.v1", config={}, source={},
                     defaults=dict(collision_policy=audit.DUAL_CONTACT_POLICY), cells=originals)
    path = tmp_path / "original.json"
    path.write_text(json.dumps(reference))
    model_id = f"{profile}_multitask_joint6_3000"
    plan = deepcopy(reference)
    plan.update(planned_episodes=300, models=1, cells_count=3, evaluation_scope=dict(
        kind="joint6_multitask_model", model_id=model_id, reference_plan_file=str(path),
        reference_plan_sha256=audit.file_sha(str(path)),
        reference_cells={color: f"{profile}_multitask_{color}" for color in audit.COLORS}))
    plan["cells"] = [dict(cell_id=f"{model_id}_{color}", model_id=model_id, model_type="multitask",
        spacing_profile=profile, target_color=color, instruction=f"Press the {color} button.",
        episodes=deepcopy(originals[index]["episodes"]), model_contract=contract(profile))
        for index, color in enumerate(audit.COLORS)]
    return plan


@pytest.mark.parametrize("profile", audit.PROFILES)
def test_multitask_scope_uses_all_three_original_color_cells(tmp_path, profile):
    plan = scoped_plan(tmp_path, profile)
    result = audit.audit_joint6_multitask_plan_scope(plan)
    assert result["reference_cells"] == {color: f"{profile}_multitask_{color}" for color in audit.COLORS}


@pytest.mark.parametrize("corruption,match", [
    ("missing_color", "three distinct"), ("duplicate_color", "three distinct"),
    ("changed_seed", "100 original"), ("missing_episode", "100 original"),
    ("wrong_instruction", "instruction"), ("wrong_spacing", "spacing_profile"),
    ("changed_checkpoint", "same frozen"), ("changed_steps", "same frozen"),
    ("wrong_reference", "reference cells"), ("wrong_reference_sha", "reference plan changed"),
    ("changed_protocol", "defaults"), ("wrong_count", "300 episodes"),
    ("wrong_model", "model_id"), ("wrong_training_group", "source_dataset_group"),
    ("wrong_training_count", "source_episode_count"), ("missing_task_contract", "Incomplete"),
])
def test_multitask_scope_rejects_changed_tasks_models_or_protocol(tmp_path, corruption, match):
    plan = scoped_plan(tmp_path)
    cell = plan["cells"][1]
    if corruption == "missing_color": plan["cells"].pop()
    elif corruption == "duplicate_color": cell["target_color"] = "red"
    elif corruption == "changed_seed": cell["episodes"][0]["seed"] += 1
    elif corruption == "missing_episode": cell["episodes"].pop()
    elif corruption == "wrong_instruction": cell["instruction"] = "Press the red button."
    elif corruption == "wrong_spacing": cell["spacing_profile"] = "far"
    elif corruption == "changed_checkpoint": cell["model_contract"]["model_sha256"] = "b" * 64
    elif corruption == "changed_steps": cell["model_contract"]["best_step"] += 1
    elif corruption == "wrong_reference": plan["evaluation_scope"]["reference_cells"]["red"] = "far_multitask_red"
    elif corruption == "wrong_reference_sha": plan["evaluation_scope"]["reference_plan_sha256"] = "0" * 64
    elif corruption == "changed_protocol": plan["defaults"]["smoothing"] = dict(mode="ema")
    elif corruption == "wrong_count": plan["planned_episodes"] = 100
    elif corruption == "wrong_model": cell["model_id"] = "medium_multitask_joint6_3000"
    elif corruption == "wrong_training_group": cell["model_contract"]["source_dataset_group"] = "near_red"
    elif corruption == "wrong_training_count": cell["model_contract"]["source_episode_count"] = 1000
    else: cell["model_contract"].pop("allowed_instructions")
    with pytest.raises(ValueError, match=match):
        audit.audit_joint6_multitask_plan_scope(plan)


def native_records(profile, color):
    records = fixtures.joint6_records()
    model = records[-1]
    model.update(identity(profile))
    cell = records[-3]
    cell.update(model_id=model["model_id"], model_type="multitask", spacing_profile=profile,
                target_color=color, instruction=f"Press the {color} button.", model_contract=contract(profile))
    records[1][0].update(model=deepcopy(model), instruction=cell["instruction"])
    return records


@pytest.mark.parametrize("profile", audit.PROFILES)
@pytest.mark.parametrize("color", audit.COLORS)
def test_multitask_native_actions_and_fixed_gripper_cover_all_nine_tasks(profile, color):
    result = fixtures.evaluate(native_records(profile, color))
    assert result["native_action_audit"]["native_action_dim"] == 6
    assert result["native_action_audit"]["verified_native_noise_samples"] == 1


@pytest.mark.parametrize("field,value", [
    ("source_dataset_group", "near_red"), ("source_episode_count", 1000),
    ("task_type", "single"), ("allowed_instructions", ["Press the red button."]),
])
def test_multitask_native_evidence_rejects_different_training_contract(field, value):
    records = native_records("near", "blue")
    records[-1][field] = value
    with pytest.raises(ValueError, match=field):
        fixtures.evaluate(records)


def test_multitask_controller_accepts_explicit_scoped_model_selector(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("multitask_eval_suite", ROOT / "scripts/run_table_buttons_policy_eval_suite.py")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    # Scope validation is exercised above; this tests the controller branch
    # without replacing its model-selector or episode-pairing checks.
    import audit_table_buttons_policy_eval as imported_audit
    monkeypatch.setattr(imported_audit, "audit_plan", audit.audit_joint6_multitask_plan_scope)
    plan = scoped_plan(tmp_path)
    for cell in plan["cells"]:
        for row in cell["episodes"]:
            row.update(layout_sha256=f"layout-{row['seed']}", effective_config_sha256=f"scene-{row['seed']}")
    reference_path = Path(plan["evaluation_scope"]["reference_plan_file"])
    reference = json.loads(reference_path.read_text())
    for original, cell in zip(reference["cells"], plan["cells"]):
        original["episodes"] = deepcopy(cell["episodes"])
    reference_path.write_text(json.dumps(reference))
    audit.file_sha.cache_clear()
    plan["evaluation_scope"]["reference_plan_sha256"] = audit.file_sha(str(reference_path))
    suite.validate_plan(plan, verify_assets=False)
