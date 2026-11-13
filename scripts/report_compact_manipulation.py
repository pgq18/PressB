#!/usr/bin/env python3
"""Audit completed raw demonstrations and replan matched legacy targets on CPU.

Example: python scripts/report_compact_manipulation.py --raw STACK_RAW
    --raw CLEANUP_RAW --output outputs/compact_manipulation_20261112/report

Writes report.json and one cases/*.json per completed physical/planning attempt.
No simulator, camera, GPU, robot, dataset filtering or video editing is used.
Legacy results are planned trajectories, NOT measured legacy success rates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pressb.runtime.demonstrations import perturb_plan
from pressb.runtime.trajectory_metrics import measured_motion_metrics, plan_motion_metrics


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    values = np.asarray(values, dtype=float)
    return (dict(count=len(values), mean=float(values.mean()), minimum=float(values.min()),
                 median=float(np.median(values)), maximum=float(values.max())) if len(values)
            else dict(count=0, mean=None, minimum=None, median=None, maximum=None))


def pose_origin(plan, settled):
    """Recover the actual clone's base origin from saved matching object poses."""
    poses = plan.get("measured_object_poses_base")
    if not poses:
        raise ValueError("No measured settled object poses; cannot claim a matched layout")
    offsets = []
    for name, pose in poses.items():
        offsets.append(np.asarray(settled["objects"][name]["position_world"], float)
                       - np.asarray(pose["position_base_m"], float))
    offsets = np.asarray(offsets)
    if offsets.shape != (len(poses), 3) or not np.isfinite(offsets).all():
        raise ValueError("Invalid saved world/base object positions")
    if np.max(np.abs(offsets-offsets[0])) > 1e-6:
        raise ValueError("Saved objects disagree about the robot base origin")
    return offsets[0]


def matched_legacy(module, config, plan, metadata, settled, runtime):
    """Use recorded settled poses, initial target and reported planner settings."""
    config = deepcopy(config)
    initial = plan["expected_initial_state"]
    config["scene"]["initial_state_joint_rad_m"] = deepcopy(initial)
    recorded_urdf_hash = runtime.get("asset_sha256", {}).get("robot_urdf")
    if recorded_urdf_hash and digest(config["scene"]["robot_urdf"]) != recorded_urdf_hash:
        raise ValueError("Current kinematic URDF differs from the recorded runtime")
    mount = float(config["scene"].get("robot_base_height_above_table_m", 0.))
    if abs(mount-float(plan["robot_base_height_above_table_m"])) > 1e-8:
        raise ValueError("Current scene mount height differs from the recorded plan")
    options = dict(trajectory_profile="legacy",
                   max_joint_speed_deg_s=plan["max_joint_speed_deg_s"])
    if metadata["task_id"] == "cube_stack":
        for name, key in (("red", "source"), ("blue", "target")):
            if not np.allclose(plan[key]["size"], config["cubes"][name]["edge_m"], rtol=0, atol=1e-8):
                raise ValueError("Current cube dimensions differ from the recorded plan")
        options["lift_height_above_table"] = plan["lift_height_above_table_m"]
    else:
        if config["bin"] != plan["bin"]:
            raise ValueError("Current bin configuration differs from the recorded plan")
        for kind in plan["kinds"]:
            spec = next(x for x in config["asset_specs"] if x["kind"] == kind)
            if not np.allclose(spec["size"], plan["grasps"][kind]["asset_size_m"], rtol=0, atol=1e-8):
                raise ValueError("Current object dimensions differ from the recorded plan")
    origin = pose_origin(plan, settled)
    legacy = module.plan_episode(config, metadata["reset_spec"], settled, origin, **options)
    # Match the noise seed/amplitude and float32 dispatch convention. The legacy
    # phase whitelist differs by design and does not change its frame count.
    noise = plan.get("noise", {})
    action_contract = runtime.get("action", {})
    bounds = {}
    if "lower" in action_contract and "upper" in action_contract:
        bounds = dict(joint_lower_rad=action_contract["lower"][:6],
                      joint_upper_rad=action_contract["upper"][:6])
    legacy = perturb_plan(legacy, int(noise.get("seed", metadata["seed"])),
                          amplitude_deg=float(noise.get("amplitude_deg", 0.)), **bounds)
    legacy["actions"] = np.asarray(legacy["actions"], dtype=np.float32).tolist()
    current_metrics = plan_motion_metrics(plan)
    legacy_metrics = plan_motion_metrics(legacy)
    return dict(status="planned", physical_execution=False,
                description="Matched settled layout, initial target and joint speed; current legacy planning profile",
                base_origin_world_m=origin.tolist(), parameters=options,
                unrecorded_planner_settings="Collection defaults for Cartesian/gripper speed and grasp/release heights",
                legacy_metrics=legacy_metrics, legacy_noise=legacy["noise"],
                current_planned_duration_s=current_metrics["duration_s"],
                legacy_planned_duration_s=legacy_metrics["duration_s"],
                planned_seconds_saved=legacy_metrics["duration_s"]-current_metrics["duration_s"],
                planned_duration_reduction_fraction=1-current_metrics["duration_s"]/legacy_metrics["duration_s"])


def inspect_physics(path):
    result = dict(rows=0, recorded_motion_rows=0, validation_rows=0,
                  first_success_action_index=None, final_phase=None, final_action_index=None,
                  phase_frames={}, final_terminated=False, final_truncated=False)
    phases = Counter()
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            result["rows"] += 1
            validating = row.get("recorded") is False or row["phase"] == "validation"
            result["validation_rows" if validating else "recorded_motion_rows"] += 1
            phases[row["phase"]] += 1
            if result["first_success_action_index"] is None and row.get("info", {}).get("success") is True:
                result["first_success_action_index"] = row["action_index"]
            result.update(final_phase=row["phase"], final_action_index=row["action_index"],
                          final_terminated=row.get("terminated", False), final_truncated=row.get("truncated", False))
    result["phase_frames"] = dict(phases)
    return result


def physical_case(directory, disposition, module, config, runtime):
    metadata = read(directory / "metadata.json")
    plan = read(directory / "plan.json")
    settled = read(directory / "settled.json")
    result = dict(kind="physical_execution", disposition=disposition,
                  directory=str(directory), attempt_index=metadata["attempt_index"],
                  episode_index=metadata["episode_index"], seed=metadata["seed"],
                  accepted=disposition == "accepted", task_id=metadata["task_id"],
                  trajectory_profile=plan.get("trajectory_profile", "legacy"),
                  physical_audit=metadata.get("audit", {}), noise=plan.get("noise"),
                  provenance={name: digest(directory/name) for name in
                              ("metadata.json", "plan.json", "settled.json", "data.npz")},
                  planned_targets=plan_motion_metrics(plan),
                  physics=inspect_physics(directory / "physics.jsonl"))
    with np.load(directory / "data.npz") as data:
        state, action = data["state"], data["action"]
        if len(state) != len(action) or len(state) < 2:
            raise ValueError("Recorded trajectory must contain aligned state/action and terminal rows")
        if not np.array_equal(action[-1], action[-2]):
            raise ValueError("Terminal target does not repeat the previous executed target")
        np.testing.assert_allclose(data["sim_time"], np.arange(len(state))/plan["fps"], rtol=0, atol=1e-10)
        executed = len(action)-1
        if executed > len(plan["actions"]):
            raise ValueError("Recorded motion exceeds planned targets")
        np.testing.assert_array_equal(action[:-1], np.asarray(plan["actions"][:executed]))
        if result["physics"]["recorded_motion_rows"] != executed:
            raise ValueError("Physical recorded-motion rows disagree with the 30 Hz sample stream")
        result["measured_states"] = measured_motion_metrics(state, plan["fps"])
        result["executed_targets"] = plan_motion_metrics(dict(
            actions=action[:-1], phases=plan["phases"][:executed], fps=plan["fps"]))
        result["alignment"] = dict(passed=True, executed_intervals=executed,
                                    recorded_frames=len(state), validation_frames_in_training=0,
                                    terminal_action="repeat_last_applied_target", row_filtering=False, retimed=False)
    if bool(metadata.get("success")) != result["accepted"]:
        raise ValueError("Folder disposition disagrees with saved physical acceptance")
    try:
        result["matched_legacy"] = matched_legacy(module, config, plan, metadata, settled, runtime)
    except Exception as exc:
        result["matched_legacy"] = dict(status="unavailable", physical_execution=False,
                                        error=f"{type(exc).__name__}: {exc}")
    return result


def aggregate(cases):
    physical = [x for x in cases if x.get("kind") == "physical_execution"]
    accepted = [x for x in physical if x["accepted"]]
    rejected = [x for x in physical if not x["accepted"]]
    planning = [x for x in cases if x.get("kind") == "planning_rejection"]
    paired = [x["matched_legacy"] for x in physical if x["matched_legacy"]["status"] == "planned"]
    resolved = len(physical)+len(planning)
    result = dict(completed_physical_attempts=len(physical), accepted_episodes=len(accepted),
                  rejected_physical_attempts=len(rejected), planning_rejections=len(planning),
                  physical_acceptance_fraction=len(accepted)/len(physical) if physical else None,
                  physical_acceptance_denominator="accepted plus rejected physically executed trajectories",
                  resolved_attempt_yield=len(accepted)/resolved if resolved else None,
                  resolved_attempt_denominator="accepted plus physical rejections plus planning rejections",
                  legacy_measured_success_rate=None,
                  legacy_comparison="CPU plans only; no legacy physical success result is inferred")
    for group, members in (("all_completed_physical", physical), ("accepted", accepted), ("rejected", rejected)):
        result[group] = dict(
            planned_duration_s=summary([x["planned_targets"]["duration_s"] for x in members]),
            recorded_duration_s=summary([x["measured_states"]["duration_s"] for x in members]),
            explicit_hold_seconds=summary([x["planned_targets"]["planned_hold_seconds"] for x in members]),
            exact_duplicate_command_count=summary([x["executed_targets"]["exact_duplicate_count"] for x in members]),
            measured_near_static_interval_fraction=summary([x["measured_states"]["near_static_interval_fraction"] for x in members]),
            max_commanded_gripper_width_m=summary([x["planned_targets"]["max_gripper_width_m"] for x in members]))
    result["matched_legacy_plans"] = dict(
        paired_count=len(paired), unavailable_count=len(physical)-len(paired),
        current_duration_s=summary([x["current_planned_duration_s"] for x in paired]),
        legacy_duration_s=summary([x["legacy_planned_duration_s"] for x in paired]),
        seconds_saved=summary([x["planned_seconds_saved"] for x in paired]),
        duration_reduction_fraction=summary([x["planned_duration_reduction_fraction"] for x in paired]))
    return result


def inspect_raw(raw, output, number):
    raw = raw.expanduser().resolve()
    manifest = read(raw / "manifest.json")
    task = manifest["task_id"]
    if task not in ("cube_stack", "tabletop_cleanup"):
        raise ValueError(f"Unsupported task {task}")
    module = importlib.import_module(f"pressb.tasks.{task}.collection")
    configuration = importlib.import_module(f"pressb.tasks.{task}.configuration")
    config = configuration.load_config(manifest["config"])
    runtime = read(raw / "runtime_metadata.json") if (raw / "runtime_metadata.json").exists() else {}
    cases, incomplete, errors = [], [], []
    for name, disposition in (("episodes", "accepted"), ("rejected", "rejected")):
        for path in sorted((raw/name).glob("*")):
            try:
                if path.is_dir():
                    if not (path/"metadata.json").exists():
                        incomplete.append(str(path)); continue
                    case = physical_case(path, disposition, module, config, runtime)
                elif name == "rejected" and path.suffix == ".json":
                    rejection = read(path)
                    case = dict(kind="planning_rejection", disposition="rejected", task_id=task,
                                directory=str(path), provenance={"rejection_sha256": digest(path)}, **rejection)
                else:
                    continue
                identifier = f"{number:02d}_{task}_attempt_{case['attempt_index']:06d}"
                target = output / "cases" / (identifier + ".json")
                write(target, case)
                case["report_path"] = str(target.resolve())
                cases.append(case)
            except Exception as exc:
                errors.append(dict(path=str(path), error=f"{type(exc).__name__}: {exc}"))
    incomplete.extend(str(p) for p in sorted((raw/"attempts").glob("*")) if p.is_dir())
    totals = aggregate(cases)
    totals["all_discovered_completed_cases_audited"] = not errors
    if errors:
        # Do not quietly remove malformed failures from a success denominator.
        totals["physical_acceptance_fraction"] = None
        totals["resolved_attempt_yield"] = None
        totals["fraction_unavailable_reason"] = "Some completed case records failed audit; see audit_errors"
    return dict(raw=str(raw), task_id=task, manifest_sha256=digest(raw/"manifest.json"),
                source_progress=read(raw/"progress.json") if (raw/"progress.json").exists() else None,
                trajectory_profile=manifest.get("trajectory_profile", "legacy"),
                summary=totals, incomplete_or_running_attempts=incomplete, audit_errors=errors,
                case_reports=[dict(kind=x["kind"], attempt_index=x["attempt_index"],
                                   report_path=x["report_path"]) for x in cases])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Directory for report.json and cases/*.json")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
                  simulation_launched=False, hardware_access=False, raw_data_modified=False,
                  script_sha256=digest(__file__),
                  limitations=["Physical acceptance is conditional on executed plans; planning rejects are also shown separately.",
                               "Near-static includes slow nonzero motion at <0.05 degree/joint and <0.1 mm gripper width per interval.",
                               "Command duplicates and measured near-static motion are different statistics.",
                               "Matched legacy paths are current CPU replans, not measured legacy rollouts or archived historical planners.",
                               "Incomplete attempts are not added to either completed denominator."],
                  runs=[inspect_raw(raw, output, index) for index, raw in enumerate(args.raw)])
    write(output/"report.json", report)
    print(json.dumps(dict(report=str(output/"report.json"),
                          summaries=[dict(task_id=r["task_id"], **r["summary"]) for r in report["runs"]]), allow_nan=False))
    return 1 if any(run["audit_errors"] for run in report["runs"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
