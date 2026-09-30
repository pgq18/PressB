#!/usr/bin/env python3
"""Separate raw model error, controller projection and physical tracking.

Reads completed evaluation recordings only. This does not import the live
policy, its IK controller, or Isaac; FK uses the independent audit implementation.
The diagnosed goal coordinates never enter policy execution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_lerobot import IndependentFK, quaternion_matrices
from audit_policy_eval import (PANEL_FIELDS, outcome_statistics, panel_episode_geometry,
                               policy_identity, smoothing_settings)

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def summary(values, scale=1.):
    values = np.asarray(values, dtype=float) * scale
    if not values.size:
        return dict(count=0, min=None, median=None, mean=None, p95=None, max=None)
    if not np.isfinite(values).all():
        raise ValueError("Diagnostic values must be finite")
    return dict(count=len(values), min=float(values.min()), median=float(np.median(values)),
                mean=float(values.mean()), p95=float(np.quantile(values, .95)), max=float(values.max()))


def tip_from_pose8(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 8 or not np.isfinite(values).all():
        raise ValueError("Expected finite pose8 action rows")
    q = values[:, 3:7]
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if not np.allclose(norms, 1., atol=1e-5, rtol=0):
        raise ValueError("Action quaternion is not unit wxyz")
    return values[:, :3] + quaternion_matrices(q / norms) @ np.array([0., 0., .24 - .1358])


def derivative_metrics(values, physics_hz, position_unit):
    """Unpadded finite differences of the recorded 120 Hz samples."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Motion metrics require finite sample-by-coordinate arrays')
    metrics = {}
    for order, name in enumerate(('velocity', 'acceleration', 'jerk'), 1):
        values = np.diff(values, axis=0) * physics_hz
        metrics[name] = dict(unit=f'{position_unit}/s' + (f'^{order}' if order > 1 else ''),
                             samples=len(values), mean_abs=float(np.abs(values).mean()) if values.size else None,
                             rms=float(np.sqrt(np.mean(values ** 2))) if values.size else None,
                             peak_abs=float(np.abs(values).max()) if values.size else None)
    return metrics


def motion_smoothness(physics, physics_hz=120):
    return dict(physics_hz=physics_hz,
        unsmoothed_joint_command=derivative_metrics(physics.get('q_command_unsmoothed', physics['q_command']), physics_hz, 'rad'),
        executed_joint_command=derivative_metrics(physics['q_command'], physics_hz, 'rad'),
        actual_tcp_position=derivative_metrics(physics['state'][:, :3], physics_hz, 'm'),
        aggregation='mean_abs/rms/peak_abs across all recorded time-coordinate values; no padding or extrapolation',
        comparison_scope='Unsmoothed and executed commands describe this same recorded stream. Different closed-loop rollouts are not controlled causal comparisons.')


def diagnose_episode(folder, last_seconds, collection=None):
    metadata = json.loads((folder / "metadata.json").read_text())
    cfg, floor = metadata["config"], int(metadata["floor"])
    with np.load(folder / "physics.npz", allow_pickle=False) as source:
        physics = {key: source[key].copy() for key in source.files}
    actions = read_jsonl(folder / "actions.jsonl")
    requests = read_jsonl(folder / "requests.jsonl")
    fk = IndependentFK(ROOT / cfg["robot_urdf"])
    base = np.asarray(metadata["robot_base_world_m"])
    panel_geometry = panel_episode_geometry(metadata, collection)
    goal = np.asarray(panel_geometry['target_face_base_m'])
    late_start = max(0., float(physics["sim_time"][-1]) - last_seconds)
    result = dict(episode=folder.name, episode_id=metadata['episode_id'], repeat=metadata['repeat'],
        floor=floor, termination=metadata["termination"], task_success=metadata['task_success'],
        **{key: panel_geometry[key] for key in PANEL_FIELDS}, panel_geometry=panel_geometry,
        success=metadata["success"], events=metadata["events"], unexpected_collisions=len(metadata["unexpected_collisions"]),
        button_face_center_base_m=goal.tolist(), sim_seconds=float(physics["sim_time"][-1]),
        late_window_start_s=late_start, late_window_end_s=float(physics["sim_time"][-1]),
        executed_actions=len(actions), groups={},
        physical_closest_distance_mm=float(physics["target_tip_distance_m"].min() * 1000),
        physical_closest_time_s=float(physics["sim_time"][physics["target_tip_distance_m"].argmin()]),
        physical_final_distance_mm=float(physics["target_tip_distance_m"][-1] * 1000),
        max_target_contact_force_n=float(physics["contact_force"][:, floor - 24].max()),
        max_any_button_contact_force_n=float(physics["contact_force"].max()),
        max_target_travel_mm=float(physics["button_travel"][:, floor - 24].max() * 1000),
        motion_smoothing=smoothing_settings(metadata), motion_smoothness=motion_smoothness(physics),
        sources={name: sha(folder / name) for name in ("metadata.json", "physics.npz", "actions.jsonl", "requests.jsonl")})
    if not actions:
        result["diagnostic_note"] = "No model action executed; inspect termination/error in metadata."
        return result
    raw8 = np.asarray([action["action_pose8"] for action in actions])
    targets = np.asarray([action["q_target"] for action in actions])
    ends = np.asarray([action["physics_end_index"] for action in actions], dtype=int)
    times = ends / 120
    raw_tip = tip_from_pose8(raw8)
    command_fk = fk.batch(targets)
    command_tip = command_fk[:, :3, 3] + command_fk[:, :3, :3] @ np.array([0., 0., .24 - .1358])
    executed_fk = fk.batch(physics['q_command'][ends])
    executed_tip = executed_fk[:, :3, 3] + executed_fk[:, :3, :3] @ np.array([0., 0., .24 - .1358])
    unsmoothed_fk = fk.batch(physics.get('q_command_unsmoothed', physics['q_command'])[ends])
    actual_tcp = physics["state"][ends, :3]
    actual_tip = physics["tip_position_world"][ends] - base
    for name, mask in (("whole_run", np.ones(len(actions), dtype=bool)), ("late_window", times >= late_start)):
        group = {}
        for label, tip in (("raw_model", raw_tip), ("ik_endpoint", command_tip),
                           ("executed_endpoint", executed_tip), ("actual_endpoint", actual_tip)):
            group[label] = dict(target_tip_distance_mm=summary(np.linalg.norm(tip[mask] - goal, axis=1), 1000),
                x_shortfall_to_button_plane_mm=summary(goal[0] - tip[mask, 0], 1000),
                yz_alignment_error_mm=summary(np.linalg.norm(tip[mask, 1:] - goal[1:], axis=1), 1000),
                y_signed_error_mm=summary(tip[mask, 1] - goal[1], 1000),
                z_signed_error_mm=summary(tip[mask, 2] - goal[2], 1000))
        group.update(ik_tcp_projection_residual_mm=summary(np.linalg.norm(command_fk[mask, :3, 3] - raw8[mask, :3], axis=1), 1000),
            actual_vs_command_tcp_error_mm=summary(np.linalg.norm(actual_tcp[mask] - command_fk[mask, :3, 3], axis=1), 1000),
            actual_vs_executed_tcp_error_mm=summary(np.linalg.norm(actual_tcp[mask] - executed_fk[mask, :3, 3], axis=1), 1000),
            executed_vs_raw_tcp_error_mm=summary(np.linalg.norm(executed_fk[mask, :3, 3] - raw8[mask, :3], axis=1), 1000),
            smoothing_tcp_displacement_mm=summary(np.linalg.norm(executed_fk[mask, :3, 3] - unsmoothed_fk[mask, :3, 3], axis=1), 1000),
            partial_interpolation_tcp_difference_mm=summary(np.linalg.norm(unsmoothed_fk[mask, :3, 3] - command_fk[mask, :3, 3], axis=1), 1000),
            actual_vs_raw_tcp_error_mm=summary(np.linalg.norm(actual_tcp[mask] - raw8[mask, :3], axis=1), 1000),
            raw_vs_ik_tip_error_mm=summary(np.linalg.norm(raw_tip[mask] - command_tip[mask], axis=1), 1000),
            velocity_limited_actions=sum(bool(actions[i]["velocity_saturated_joints"]) for i in np.flatnonzero(mask)),
            action_count=int(mask.sum()))
        result["groups"][name] = group
    physical_mask = physics["sim_time"] >= late_start
    result["late_window_physical_tip_x_slope_mm_per_s"] = (
        float(np.polyfit(physics["sim_time"][physical_mask],
              (physics["tip_position_world"][physical_mask] - base)[:, 0], 1)[0] * 1000)
        if physical_mask.sum() > 1 else None)
    first, last, skipped = [], [], []
    for request in requests:
        if request["observation_sim_time"] < late_start:
            continue
        try:
            predicted_tip = tip_from_pose8(request["response"]["actions_pose8"])
        except (ValueError, KeyError) as exc:
            skipped.append(dict(chunk_index=request["chunk_index"], error=str(exc)))
            continue
        observed_tip = physics["tip_position_world"][request["observation_physics_index"]] - base
        first.append(predicted_tip[0, 0] - observed_tip[0])
        last.append(predicted_tip[-1, 0] - observed_tip[0])
    result["late_window_model_first_row_forward_delta_from_observed_tip_mm"] = summary(first, 1000)
    result["late_window_model_last_row_forward_delta_from_observed_tip_mm"] = summary(last, 1000)
    result["skipped_invalid_requests"] = skipped
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--last-seconds", type=float, default=5.)
    args = parser.parse_args()
    if not np.isfinite(args.last_seconds) or args.last_seconds <= 0:
        parser.error("last-seconds must be finite and positive")
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((run / "eval_manifest.json").read_text())
    collection = json.loads((Path(manifest['arguments']['dataset']) / 'meta/collection_metadata.json').read_text())
    identity = policy_identity(manifest, json.loads((run / "policy_service.json").read_text()))
    smoothing = smoothing_settings(manifest)
    report_path = run / "report.json"
    run_report = json.loads(report_path.read_text()) if report_path.exists() else {}
    folders = sorted(path.parent for path in run.glob("episode_*/metadata.json"))
    if not folders:
        raise ValueError("No completed episodes found")
    episodes = [diagnose_episode(folder, args.last_seconds, collection) for folder in folders]
    rows = []
    for episode in episodes:
        late = episode["groups"].get("late_window")
        if late is None:
            continue
        rows.append(dict(floor=episode["floor"], episode=episode["episode"], success=episode["success"],
            episode_id=episode['episode_id'], repeat=episode['repeat'],
            **{key: episode[key] for key in PANEL_FIELDS},
            raw_tip_x_shortfall_median_mm=late["raw_model"]["x_shortfall_to_button_plane_mm"]["median"],
            actual_tip_x_shortfall_median_mm=late["actual_endpoint"]["x_shortfall_to_button_plane_mm"]["median"],
            raw_tip_yz_error_median_mm=late["raw_model"]["yz_alignment_error_mm"]["median"],
            raw_tip_z_signed_error_median_mm=late["raw_model"]["z_signed_error_mm"]["median"],
            ik_tcp_projection_median_mm=late["ik_tcp_projection_residual_mm"]["median"],
            ik_tcp_projection_p95_mm=late["ik_tcp_projection_residual_mm"]["p95"],
            tracking_tcp_median_mm=late["actual_vs_executed_tcp_error_mm"]["median"],
            smoothing_tcp_displacement_median_mm=late["smoothing_tcp_displacement_mm"]["median"],
            executed_tcp_residual_median_mm=late["executed_vs_raw_tcp_error_mm"]["median"],
            velocity_limited_actions=late["velocity_limited_actions"], action_count=late["action_count"],
            closest_actual_tip_distance_mm=episode["physical_closest_distance_mm"],
            max_any_button_contact_force_n=episode["max_any_button_contact_force_n"]))
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), run=str(run),
        run_complete=run_report.get("complete", False), diagnosed_episodes=len(episodes), late_window_seconds=args.last_seconds,
        diagnostic_source_sha256=sha(Path(__file__)), independent_fk_source_sha256=sha(ROOT / "scripts/audit_lerobot.py"),
        **identity, policy_service_sha256=sha(run / "policy_service.json"), eval_manifest_sha256=sha(run / "eval_manifest.json"),
        motion_smoothing=smoothing, outcomes=outcome_statistics(episodes),
        raw_target_tip_formula="raw_tcp_xyz + R_from_normalized_wxyz @ [0,0,0.24-0.1358], base_link frame",
        limitations=["Diagnostic decomposition is not a physical success audit; inspect audit.json separately.",
                     "Measured panel corner projection proves frustum containment, not absence of occlusion.",
                     "IK residual evaluates the requested full30Hz endpoint; a terminal partial interval may stop before that endpoint.",
                     "actual_vs_command_tcp_error_mm is the legacy actual-versus-full-IK-endpoint metric; tracking_tcp_median_mm now uses actual_vs_executed_tcp_error_mm.",
                     "Smoothing displacement compares sent and unsmoothed commands at the same physical endpoint, including terminal partial intervals.",
                     "Joint-command and actual-TCP derivatives describe recorded streams; different policy rollouts do not isolate a causal smoothing effect.",
                     "Pose matching at one time does not establish eventual closed-loop success."],
        summary=rows, episodes=episodes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(output), run_complete=result["run_complete"], summary=rows), indent=2))


if __name__ == "__main__":
    main()
