"""Aggregation completeness and evidence binding using tiny pre-audited fixtures."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from aggregate_policy_eval import aggregate, LAYOUTS, PANEL_FIELDS, sha, write_outputs


MODEL_SHA = "a" * 64
SMOOTHING = dict(kind="linear_joint_interpolation_then_causal_mean", window=3, physics_hz=120,
                 nominal_delay_s=1/120, initial_history="repeat_initial_command", reset="per_episode", history="cross_chunk")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_worker(root, worker_index, floors, *, seed=20260930):
    worker = root / f"worker_{worker_index}"
    worker.mkdir()
    cfg = {"physics_dt": 1/120, "panel_randomization": {"enabled": True}}
    snapshot = root / "scene.usda"
    snapshot.write_text("#usda 1.0\n")
    identity = dict(config=cfg, scene_sha256=sha(snapshot))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    collection = dict(identity=identity, collection_fingerprint=fingerprint, **identity)
    dataset = root / "dataset"
    save(dataset / "meta/collection_metadata.json", collection)
    layouts = [dict(panel_layout_index=i, panel_layout_name=name, panel_offset_x_m=x, panel_offset_y_m=y)
               for i, (name, x, y) in enumerate(LAYOUTS)]
    schedule = [dict(episode_id=i*3+j, floor=f, repeat=0, seed_episode_index=i*12+f-24, **layout)
                for i, layout in enumerate(layouts) for j, f in enumerate(floors)]
    args = dict(expected_checkpoint_sha256=MODEL_SHA, expected_checkpoint_step=10600, panel_layouts="center_corners",
                floors=list(floors), episodes_per_floor=1, max_seconds=15., smoothing_window=3, seed=seed,
                dataset=str(dataset), snapshot=str(snapshot))
    manifest = dict(arguments=args, expected_checkpoint_sha256=MODEL_SHA, expected_checkpoint_step=10600,
                    panel_layout_mode="center_corners", panel_layouts=layouts, episode_schedule=schedule,
                    motion_smoothing=SMOOTHING, recorded_actions_used=False, target_planner_used=False,
                    config=cfg, scene_sha256=sha(snapshot), collection_fingerprint=fingerprint, sources={})
    service = dict(status="ready", checkpoint_verified=True, model_sha256=MODEL_SHA, checkpoint_step=10600,
                   fps=30, action_horizon=7, camera_order=["global", "wrist"],
                   provenance=dict(model_sha256=MODEL_SHA, checkpoint="/checkpoint/step_010600"))
    save(worker / "eval_manifest.json", manifest)
    save(worker / "policy_service.json", service)
    proof = dict(checkpoint_sha256=MODEL_SHA, checkpoint_step=10600,
                 eval_manifest_sha256=sha(worker / "eval_manifest.json"), policy_service_sha256=sha(worker / "policy_service.json"),
                 scene_sha256=sha(snapshot), collection_fingerprint=fingerprint,
                 dataset_collection_sha256=sha(dataset / "meta/collection_metadata.json"),
                 panel_layout_mode="center_corners", panel_layouts=layouts)
    reported, audited, diagnosed = [], [], []
    for planned in schedule:
        directory = worker / f"episode_{planned['episode_id']:06d}"
        directory.mkdir()
        metadata = dict(planned, config=cfg, motion_smoothing=SMOOTHING)
        save(directory / "metadata.json", metadata)
        for name in ("physics.npz", "frames.npz", "global.mp4", "wrist.mp4"):
            (directory / name).write_bytes(b"small synthetic pre-audited artifact")
        (directory / "requests.jsonl").write_text("")
        (directory / "actions.jsonl").write_text("")
        files = {name: sha(directory / name) for name in ("metadata.json", "physics.npz", "frames.npz", "requests.jsonl", "actions.jsonl")}
        row = dict(planned, success=False, task_success=False, termination="time_limit")
        reported.append(row)
        audited.append(dict(row, audit_pass=True, checked_inference_base_seed=seed, files_sha256=files,
                            requests=0, videos={v: {"sha256": sha(directory / f"{v}.mp4")} for v in ("global", "wrist")},
                            simulation_seconds=15., min_target_tip_distance_m=.02, final_target_tip_distance_m=.03,
                            events=[], unexpected_collisions=0, max_command_projection_m=.01))
        diagnosed.append(dict(row, sources={k: v for k, v in files.items() if k != "frames.npz"}))
    report = dict(complete=True, total_episodes=15, passed_episodes=0, task_successes=0, fps=30, action_chunk_size=7,
                  max_sim_seconds=15., panel_layout_mode="center_corners", panel_layouts=layouts,
                  motion_smoothing=SMOOTHING, episodes=reported)
    audit = dict(run=str(worker), audit_pass=True, errors=[], audited_episodes=15, collision_free_successes=0,
                 task_successes=0, motion_smoothing=SMOOTHING, provenance=proof, audit_source_sha256="audit-code", episodes=audited)
    diagnosis = dict(run=str(worker), run_complete=True, diagnosed_episodes=15, motion_smoothing=SMOOTHING,
                     diagnostic_source_sha256="diagnosis-code", episodes=diagnosed,
                     **{k: proof[k] for k in ("checkpoint_sha256", "checkpoint_step", "eval_manifest_sha256", "policy_service_sha256")})
    for name, value in (("report.json", report), ("audit.json", audit), ("diagnosis.json", diagnosis),
                        ("status.json", {"status": "complete"})):
        save(worker / name, value)


@pytest.fixture
def completed(tmp_path):
    for i in range(4):
        fixture_worker(tmp_path, i, range(24+i*3, 27+i*3))
    return tmp_path


def change(path, fn):
    value = json.loads(path.read_text())
    fn(value)
    save(path, value)


def test_passing_audits_with_zero_success_are_reported_as_zero(completed):
    report = aggregate(completed, MODEL_SHA)
    assert report["aggregation_pass"] and report["all_worker_audits_pass"]
    assert report["overall"]["episodes"] == 60
    assert report["overall"]["successes"] == 0 and report["overall"]["success_rate"] == 0
    assert len(report["by_floor"]) == 12 and all(r["episodes"] == 5 for r in report["by_floor"].values())
    assert all(r["episodes"] == 12 for r in report["by_position"].values())
    assert len({(r["worker"], r["local_episode_id"]) for r in report["episodes"]}) == 60
    write_outputs(completed, report)
    with (completed / "episodes.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 60
    assert "0/60" in (completed / "summary.md").read_text()
    with pytest.raises(ValueError, match="already exist"):
        write_outputs(completed, report)


@pytest.mark.parametrize("kind", ["audit_fail", "unfinished", "missing_episode", "duplicate_episode", "diagnosis_incomplete",
                                  "wrong_model", "wrong_step", "success_disagreement", "changed_metadata", "changed_video",
                                  "stale_diagnosis", "changed_manifest", "incomplete_inventory"])
def test_incomplete_inconsistent_or_modified_evidence_is_rejected(completed, kind):
    worker = completed / "worker_2"
    if kind == "audit_fail":
        change(worker / "audit.json", lambda r: r.update(audit_pass=False))
    elif kind == "unfinished":
        change(worker / "report.json", lambda r: r.update(complete=False))
    elif kind == "missing_episode":
        change(worker / "report.json", lambda r: r["episodes"].pop())
    elif kind == "duplicate_episode":
        change(worker / "audit.json", lambda r: r["episodes"].__setitem__(1, deepcopy(r["episodes"][0])))
    elif kind == "diagnosis_incomplete":
        change(worker / "diagnosis.json", lambda r: r.update(run_complete=False))
    elif kind == "wrong_model":
        change(worker / "policy_service.json", lambda r: r.update(model_sha256="b"*64))
    elif kind == "wrong_step":
        change(worker / "audit.json", lambda r: r["provenance"].update(checkpoint_step=10500))
    elif kind == "success_disagreement":
        change(worker / "report.json", lambda r: r["episodes"][0].update(success=True))
    elif kind == "changed_metadata":
        (worker / "episode_000000/metadata.json").write_text("changed")
    elif kind == "changed_video":
        (worker / "episode_000000/global.mp4").write_bytes(b"changed")
    elif kind == "stale_diagnosis":
        change(worker / "diagnosis.json", lambda r: r["episodes"][0]["sources"].update({"physics.npz": "c"*64}))
    elif kind == "changed_manifest":
        change(worker / "eval_manifest.json", lambda r: r["arguments"].update(seed=1))
    else:
        change(worker / "audit.json", lambda r: r["episodes"][0]["files_sha256"].pop("frames.npz"))
    with pytest.raises(ValueError):
        aggregate(completed, MODEL_SHA)
    assert not (completed / "aggregate_report.json").exists()


def test_duplicate_floor_groups_across_otherwise_valid_workers_rejected(tmp_path):
    for i in range(4):
        fixture_worker(tmp_path, i, range(24, 27) if i == 3 else range(24+i*3, 27+i*3))
    with pytest.raises(ValueError, match="duplicate floor/layout"):
        aggregate(tmp_path, MODEL_SHA)


def test_individually_valid_but_different_worker_seed_rejected(tmp_path):
    for i in range(4):
        fixture_worker(tmp_path, i, range(24+i*3, 27+i*3), seed=20260931 if i == 3 else 20260930)
    with pytest.raises(ValueError, match="identity differs"):
        aggregate(tmp_path, MODEL_SHA)


def test_missing_worker_rejected(tmp_path):
    for i in range(3):
        fixture_worker(tmp_path, i, range(24+i*3, 27+i*3))
    with pytest.raises(ValueError, match="exactly worker"):
        aggregate(tmp_path, MODEL_SHA)
