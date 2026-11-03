import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = script("prepare_table_buttons_policy_eval")
suite = script("run_table_buttons_policy_eval_suite")


def fixture_plan(count=3):
    episodes = {profile: {color: [dict(episode_index=index, seed=10000+index,
        layout_sha256=f"{profile}-{index}", effective_config_sha256=f"{profile}-{color}-{index}")
        for index in range(count)] for color in prepare.COLORS} for profile in prepare.PROFILES}
    cells = prepare.build_cells(episodes)
    return dict(schema="pressb.table_buttons.policy_eval_plan.v1", cells=cells, planned_episodes=18*count)


def episode_files(directory, cell, episode, success, *, seed=None, latency_key="latency_s"):
    directory.mkdir(parents=True)
    metadata = dict(evaluation_status="complete", **{key: cell[key] for key in (
        "cell_id", "model_id", "spacing_profile", "target_color")}, episode_index=episode["episode_index"],
        seed=episode["seed"] if seed is None else seed, policy_seed=episode["policy_seed"],
        success=success, stop_reason="success" if success else "timeout", sim_duration_s=1. if success else 12.,
        model_sha256="a"*64)
    for name in ("metadata.json", "report.json"):
        (directory/name).write_text(json.dumps(metadata))
    for name in ("data.npz", "physics.npz", "control.jsonl", "contacts.json", "unexpected_contacts.json"):
        (directory/name).write_text("fixture")
    (directory/"policy_requests.jsonl").write_text(json.dumps({latency_key: .2})+"\n"+json.dumps({latency_key: .4})+"\n")


def test_pairing_and_rng_windows_are_shared_only_as_intended():
    plan=fixture_plan(100)
    suite.validate_plan(plan,verify_assets=False)
    for profile in prepare.PROFILES:
        for color in prepare.COLORS:
            single=next(cell for cell in plan["cells"] if cell["cell_id"]==f"{profile}_{color}")
            multi=next(cell for cell in plan["cells"] if cell["cell_id"]==f"{profile}_multitask_{color}")
            assert single["episodes"]==multi["episodes"]
    windows=sorted(prepare.policy_seed(profile,color,index) for profile in prepare.PROFILES
        for color in prepare.COLORS for index in range(100))
    assert len(set(windows))==900
    assert min(b-a for a,b in zip(windows,windows[1:]))>=1000


def test_plan_rejects_a_pair_with_different_layout_or_rng():
    plan=fixture_plan()
    plan["cells"][3]["episodes"][0]["policy_seed"]+=1
    with pytest.raises(ValueError,match="share scene and policy RNG"):
        suite.validate_plan(plan,verify_assets=False)


def test_schedule_interleaves_servers():
    cells=suite.scheduling_order(fixture_plan()["cells"])
    assert [cell["cell_id"] for cell in cells[:3]]==["near_red","medium_red","far_red"]
    assert [cell["cell_id"] for cell in cells[9:12]]==["near_multitask_red","medium_multitask_red","far_multitask_red"]


def test_missing_episode_is_not_a_task_failure_and_has_no_final_rate(tmp_path):
    cell=fixture_plan()["cells"][0]
    for index,success in ((0,True),(1,False)):
        episode_files(tmp_path/"episodes"/f"episode_{index:06d}",cell,cell["episodes"][index],success,
            latency_key="round_trip_seconds" if index==1 else "latency_s")
    summary=suite.EpisodeIndex().summarize_cell(tmp_path,cell)
    assert summary["completed"]==2 and summary["task_failures"]==1
    assert summary["missing_episode_ids"]==[2] and summary["success_rate"] is None
    assert summary["success_rate_over_completed"]==.5
    assert summary["inference_latency_s"]["count"]==4
    assert summary["inference_latency_s"]["mean"]==pytest.approx(.3)
    assert summary["simulation_duration_s"]["total"]==13.


def test_foreign_seed_or_incomplete_episode_does_not_enter_success_denominator(tmp_path):
    cell=fixture_plan()["cells"][0]
    path=tmp_path/"episodes/episode_000000"
    episode_files(path,cell,cell["episodes"][0],True,seed=99)
    summary=suite.EpisodeIndex().summarize_cell(tmp_path,cell)
    assert summary["completed"]==0 and summary["task_failures"]==0 and len(summary["invalid"])==1


def test_aggregate_model_success_uses_all_tasks(tmp_path):
    plan=fixture_plan(1)
    summaries={}
    for cell in plan["cells"]:
        directory=tmp_path/cell["cell_id"]
        episode_files(directory/"episodes/episode_000000",cell,cell["episodes"][0],cell["target_color"]!="blue")
        summaries[cell["cell_id"]]=suite.EpisodeIndex().summarize_cell(directory,cell)
    aggregate=suite.aggregate_cells(summaries)
    assert aggregate["by_model"]["near_multitask"]["success_rate"]==pytest.approx(2/3)
    assert aggregate["by_spacing"]["near"]["completed"]==6
    assert aggregate["by_color"]["blue"]["success_rate"]==0


def test_timing_rejects_nonfinite_values_instead_of_inventing_zero():
    assert suite.summarize_numbers([])["mean"] is None
    with pytest.raises(ValueError):
        suite.summarize_numbers([float("nan")])


def test_controller_preserves_display_and_physical_gpu_enumeration(monkeypatch):
    monkeypatch.setenv("DISPLAY",":1")
    monkeypatch.setenv("XAUTHORITY","/tmp/fixture.Xauthority")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES",raising=False)
    controller=object.__new__(suite.Controller)
    env=controller.environment()
    assert env["DISPLAY"]==":1" and env["XAUTHORITY"]=="/tmp/fixture.Xauthority"
    assert "CUDA_VISIBLE_DEVICES" not in env
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","1")
    with pytest.raises(ValueError,match="physical"):
        controller.environment()


def test_derived_formal_plan_keeps_parent_and_all_pairing_untouched(tmp_path):
    parent=fixture_plan(2)
    parent["config"]={"physics_dt":1/120}
    parent["defaults"]={"timeout_s":12.,"execute_steps":7}
    source=tmp_path/"plan.json"
    source.write_text(json.dumps(parent))
    original=source.read_bytes()
    derived=prepare.reuse_plan(source,tmp_path/"run","finger_closure_dual")
    assert source.read_bytes()==original
    assert derived["cells"]==parent["cells"] and derived["config"]==parent["config"]
    assert derived["defaults"]["collision_policy"]==prepare.DUAL_CONTACT_POLICY
    assert derived["parent_plan"]["sha256"]==prepare.sha256(source)
    assert derived["metrics"]["primary"]=="task_success"


def dual_episode(directory,cell,episode,closure=False):
    episode_files(directory,cell,episode,True)
    for name in ("metadata.json","report.json"):
        record=json.loads((directory/name).read_text())
        record.update(collision_policy=prepare.DUAL_CONTACT_POLICY,task_success=True,
                      strict_collector_success=not closure)
        (directory/name).write_text(json.dumps(record))
    event={"actors":["/World/Piper/link7","/World/Piper/link8"],"force_n":.03}
    (directory/"unexpected_contacts.json").write_text(json.dumps([event] if closure else []))
    (directory/"finger_closure_contacts.json").write_text(json.dumps([event] if closure else []))
    (directory/"blocking_contacts.json").write_text("[]")


def test_dual_outcomes_and_closure_rate_are_kept_separate(tmp_path):
    cell=fixture_plan(2)["cells"][0]
    for index in range(2):
        dual_episode(tmp_path/"episodes"/f"episode_{index:06d}",cell,cell["episodes"][index],closure=index==1)
    result=suite.EpisodeIndex(prepare.DUAL_CONTACT_POLICY).summarize_cell(tmp_path,cell)
    assert result["task_success_rate"]==1.
    assert result["strict_collector_success_rate"]==.5
    assert result["finger_closure_episode_rate"]==.5
    assert result["strict_collector_failures"]==1 and result["finger_closure_events"]==1
    grouped=suite.aggregate_cells({cell["cell_id"]:result})["by_model"][cell["model_id"]]
    assert grouped["task_success_rate"]==1. and grouped["strict_collector_success_rate"]==.5


def test_formal_plan_rejects_missing_dual_protocol_metrics(tmp_path):
    cell=fixture_plan(1)["cells"][0]
    episode_files(tmp_path/"episodes/episode_000000",cell,cell["episodes"][0],True)
    result=suite.EpisodeIndex(prepare.DUAL_CONTACT_POLICY).summarize_cell(tmp_path,cell)
    assert result["completed"]==0 and len(result["invalid"])==1


def test_original_metric_cannot_hide_a_logged_closure(tmp_path):
    cell=fixture_plan(1)["cells"][0]
    directory=tmp_path/"episodes/episode_000000"
    dual_episode(directory,cell,cell["episodes"][0],closure=True)
    for name in ("metadata.json","report.json"):
        record=json.loads((directory/name).read_text())
        record["strict_collector_success"]=True
        (directory/name).write_text(json.dumps(record))
    result=suite.EpisodeIndex(prepare.DUAL_CONTACT_POLICY).summarize_cell(tmp_path,cell)
    assert result["completed"]==0 and len(result["invalid"])==1
