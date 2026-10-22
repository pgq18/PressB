#!/usr/bin/env python3
"""Analyze recorded PAP evaluation outcomes without changing or rerunning episodes.

Default requires every planned report to be complete. --partial writes separate
*_partial.json files and explicitly labels all its output as provisional.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pressb.pick_place.task import object_geometry, object_inside_box


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def intervals(mask, times, minimum_seconds=0.0):
    """Contiguous measured runs; duration is between first and last samples."""
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)-1
    return [dict(start_s=float(times[a]), end_s=float(times[b]),
                 duration_s=float(times[b]-times[a]))
            for a, b in zip(starts, stops) if times[b]-times[a] >= minimum_seconds]


def analyze_failure(report, episode_dir):
    episode = report["evaluation_episode"]
    with np.load(episode_dir / "physics.npz", allow_pickle=False) as data:
        times = data["time_s"].copy()
        positions = data["cube_position"].copy()
        quaternions = data["cube_quaternion"].copy()
        velocity = data["cube_velocity"].copy()
        joints = data["q_actual"].copy()
    contacts = json.loads((episode_dir / "contacts.json").read_text())
    cfg, scene = report["config"], report["scene"]
    geometry = object_geometry(scene["object_geometry"])
    rotation = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()
    # Exact vertical support distance, also valid for the existing PAP cylinder.
    if geometry["shape"] == "cylinder":
        half_extent = (geometry["radius"]*np.linalg.norm(rotation[:, 2, :2], axis=1)
                       + geometry["height"]*.5*np.abs(rotation[:, 2, 2]))
    else:
        half_extent = np.abs(rotation[:, 2, :]) @ (np.asarray(geometry["size_xyz"])*.5)
    bottom = positions[:, 2]-half_extent
    clearance = bottom-float(cfg["table_height"])
    cavity = scene["box_inside"]
    box = dict(center_world=cavity["origin_world"], yaw_deg=cavity["yaw_deg"],
               inner_size_xy=(np.asarray(cavity["max_local"])-np.asarray(cavity["min_local"]))[:2],
               floor_z_world=cavity["origin_world"][2]+cavity["min_local"][2],
               rim_z_world=cavity["origin_world"][2]+cavity["max_local"][2])
    inside = np.array([object_inside_box(p, q, geometry, box)
                       for p, q in zip(positions, quaternions)])
    lifted = clearance >= .04
    first_lift = int(np.flatnonzero(lifted)[0]) if lifted.any() else None
    table_height_mask = np.abs(clearance) <= .003
    after_lift = np.zeros(len(times), dtype=bool)
    if first_lift is not None:
        after_lift[first_lift+1:] = True
    return_mask = table_height_mask & after_lift
    first_return = int(np.flatnonzero(return_mask)[0]) if return_mask.any() else None
    relift = (np.flatnonzero(lifted & (np.arange(len(times)) > first_return))
              if first_return is not None else np.empty(0, dtype=int))
    width = joints[:, 6]-joints[:, 7]
    final = report["physical_outcome"]
    dt = float(cfg["physics_dt"])
    contact_evidence = {}
    for finger in ("link7", "link8"):
        events = [event for event in contacts if event["finger"] == finger]
        positive = [event for event in events if event["impulse_ns"] > 1e-7]
        contact_evidence[finger] = dict(
            events=len(events),
            first_s=min((event["tick"]*dt for event in events), default=None),
            last_s=max((event["tick"]*dt for event in events), default=None),
            last_nonzero_impulse_s=max((event["tick"]*dt for event in positive), default=None))
    # Holding classification requires observed contact at the final sample and a
    # narrow measured gripper; position above the table alone does not prove it.
    held_final = (bool(final.get("finger_contact", False))
                  and float(width[-1]) < geometry["grasp_width_m"]+.008
                  and float(clearance[-1]) > .003)
    inv = Rotation.from_euler("z", -float(box["yaw_deg"]), degrees=True).as_matrix()
    local_final = inv @ (positions[-1]-np.asarray(box["center_world"]))
    above_box_xy = bool(np.all(np.abs(local_final[:2]) <= np.asarray(box["inner_size_xy"])*.5))
    final_table = bool(table_height_mask[-1] and np.linalg.norm(velocity[-1, :3]) < .025)
    if first_lift is None:
        stage = "no_valid_lift_4cm"
    elif inside.any():
        stage = "entered_box_without_successful_stable_release"
    elif held_final:
        stage = "held_above_box_at_timeout" if above_box_xy else "held_elsewhere_at_timeout"
    elif first_return is not None and final_table:
        stage = "lifted_then_returned_to_table"
    elif float(positions[-1, 2]) < float(cfg["table_height"])-.03:
        stage = "lifted_then_fell_below_table"
    else:
        stage = "lifted_without_completed_placement_other"
    geometry_stage = ("not_lifted_4cm" if first_lift is None else
                      "lifted_never_fully_inside_box" if not inside.any() else
                      "inside_box_but_no_successful_stable_release")
    return dict(
        episode_id=episode["episode_id"], color=episode["color"], pair_index=episode["pair_index"],
        initial_xy=episode["center_xy"], official_failure_reason=report["failure_reason"],
        duration_s=report["duration_s"], stage=stage, geometry_stage=geometry_stage,
        max_bottom_clearance_m=float(clearance.max()),
        max_clearance_time_s=float(times[int(clearance.argmax())]),
        first_lift_4cm_s=float(times[first_lift]) if first_lift is not None else None,
        first_return_to_tabletop_height_after_lift_s=float(times[first_return]) if first_return is not None else None,
        next_lift_4cm_after_first_return_s=float(times[relift[0]]) if len(relift) else None,
        lift_intervals=intervals(lifted, times),
        table_height_intervals_after_lift=intervals(return_mask, times, .1),
        ever_fully_inside_box=bool(inside.any()), inside_box_intervals=intervals(inside, times),
        final_held_contact_and_width_evidence=held_final, final_above_box_xy=above_box_xy,
        final_tabletop_height_and_low_linear_speed=final_table,
        final_position_world=positions[-1].tolist(), final_bottom_clearance_m=float(clearance[-1]),
        final_gripper_width_m=float(width[-1]), final_linear_speed_m_s=float(np.linalg.norm(velocity[-1, :3])),
        final_angular_speed_rad_s=float(np.linalg.norm(velocity[-1, 3:])),
        final_physical_outcome=final, contact_evidence=contact_evidence,
        contact_counts=report["contact_counts"],
        trajectory_evidence=str((episode_dir / "physics.npz").relative_to(episode_dir.parents[2])),
        images_visually_reviewed_by_this_script=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Evaluation directory containing run/")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / "run/evaluation_plan.json").read_text())
    reports, missing = {}, []
    for episode in plan["episodes"]:
        path = root / "run/episodes" / episode["episode_id"] / "report.json"
        if not path.exists():
            missing.append(episode["episode_id"])
            continue
        report = json.loads(path.read_text())
        if report.get("evaluation_status") != "complete":
            missing.append(episode["episode_id"])
            continue
        if report["evaluation_episode"] != episode:
            raise ValueError(f"Plan/report mismatch: {path}")
        reports[episode["episode_id"]] = report
    if missing and not args.partial:
        raise SystemExit(f"Evaluation incomplete: {len(reports)}/{len(plan['episodes'])}; use --partial for provisional files")
    suffix = "_partial" if args.partial else ""
    common = dict(analysis_status="partial" if args.partial else "complete",
                  planned_episodes=len(plan["episodes"]), completed_episodes=len(reports),
                  generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  region_kind=plan["region"]["region_kind"],
                  source_run="run", reruns_performed=False,
                  method="Read recorded 120 Hz measured object trajectories and contact events; preserve original success labels. Returning to tabletop height is supported by the trajectory, not inferred from final position alone. Held labels require final measured contact, narrow gripper, and elevated object. No claim of policy-error causality.")
    group_summaries = []
    for group in ("position_red", "position_color"):
        group_plan = [episode for episode in plan["episodes"] if episode["group"] == group]
        rows = [reports[episode["episode_id"]] for episode in group_plan if episode["episode_id"] in reports]
        failures = [analyze_failure(report, root / "run/episodes" / report["evaluation_episode"]["episode_id"])
                    for report in rows if not report["physical_success"]]
        summary = dict(**common, group=group, group_complete=len(rows) == len(group_plan),
                       episodes=len(rows), group_planned_episodes=len(group_plan),
                       successes=sum(bool(report["physical_success"]) for report in rows), failure_count=len(failures),
                       stage_counts=dict(Counter(item["stage"] for item in failures)),
                       geometry_stage_counts=dict(Counter(item["geometry_stage"] for item in failures)), failures=failures)
        write(root / f"failures_{group}{suffix}.json", summary)
        group_summaries.append({key: value for key, value in summary.items() if key not in ("failures", "method")})
    baseline = {report["evaluation_episode"]["pair_index"]: report for report in reports.values()
                if report["evaluation_episode"]["group"] == "position_red"}
    colors = []
    for color in ("red", "green", "blue", "white", "yellow"):
        rows = [report for report in reports.values() if report["evaluation_episode"]["group"] == "position_color"
                and report["evaluation_episode"]["color"] == color]
        pairs = []
        for report in rows:
            episode = report["evaluation_episode"]
            if episode["pair_index"] not in baseline:
                continue
            original = baseline[episode["pair_index"]]
            if episode["center_xy"] != original["evaluation_episode"]["center_xy"]:
                raise ValueError("Pair position mismatch")
            pairs.append(dict(pair_index=episode["pair_index"], center_xy=episode["center_xy"],
                              baseline_red_success=bool(original["physical_success"]),
                              random_color_success=bool(report["physical_success"])))
        counts = Counter((pair["baseline_red_success"], pair["random_color_success"]) for pair in pairs)
        colors.append(dict(color=color, completed_pairs=len(pairs),
                           baseline_red_successes=sum(pair["baseline_red_success"] for pair in pairs),
                           random_color_successes=sum(pair["random_color_success"] for pair in pairs),
                           both_success=counts[True, True], baseline_only_success=counts[True, False],
                           color_only_success=counts[False, True], both_fail=counts[False, False], pairs=pairs))
    write(root / f"paired_color_breakdown{suffix}.json", dict(
        **common, same_positions_verified=True, policy_seed_paired=False, colors=colors,
        interpretation="Each second-group color compares with the first-group red outcome at exactly the same sampled positions. Policy inference draws are not shared/seed-paired. The red subset is a same-color repeat; a difference there cannot be caused by changing color."))
    print(json.dumps(dict(groups=group_summaries,
                          paired_colors=[{key: value for key, value in row.items() if key != "pairs"} for row in colors]), indent=2))


if __name__ == "__main__":
    main()
