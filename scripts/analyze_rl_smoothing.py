#!/usr/bin/env python3
"""Compare measured rollout smoothness without changing or retiming trajectories.

Only PhysX joint velocities and measured TCP states enter motion metrics. Every
episode is measured over its real lifetime, with no terminal padding. Comparisons
use both all matched tasks and the common-success subset, so a failing policy
cannot appear smoother solely because it stops early.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
METRIC_UNITS = {
    "duration_s": "s",
    "joint_speed_rms": "rad/s",
    "joint_acceleration_rms": "rad/s^2",
    "joint_jerk_rms": "rad/s^3",
    "tcp_speed_rms": "m/s",
    "tcp_acceleration_rms": "m/s^2",
    "tcp_jerk_rms": "m/s^3",
    "tcp_angular_speed_rms": "rad/s",
    "tcp_angular_acceleration_rms": "rad/s^2",
    "tcp_path_length_m": "m",
    "joint_path_length_rad": "rad",
}


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def derivative(values, times):
    """First divided differences, assigned to true interval midpoints."""
    values, times = np.asarray(values, dtype=float), np.asarray(times, dtype=float)
    require(times.ndim == 1 and len(values) == len(times), "Mismatched samples and times")
    require(np.isfinite(values).all() and np.isfinite(times).all(), "Nonfinite trajectory")
    require(np.all(np.diff(times) > 0), "Timestamps must strictly increase")
    shape = (-1,) + (1,) * (values.ndim - 1)
    return np.diff(values, axis=0) / np.diff(times).reshape(shape), (times[1:] + times[:-1]) / 2


def vector_rms(values):
    """RMS Euclidean magnitude, not mean of separately normalized dimensions."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return float(np.sqrt(np.mean(np.sum(values * values, axis=-1))))


def arm_indices(joint_names):
    names = list(joint_names)
    require(len(set(names)) == len(names), "Repeated joint names")
    require(set(ARM_JOINT_NAMES).issubset(names), "Cannot identify all six arm joints")
    return [names.index(name) for name in ARM_JOINT_NAMES]


def normalize_quaternions(quaternions):
    q = np.asarray(quaternions, dtype=float)
    require(q.ndim == 2 and q.shape[1] == 4, "Quaternion array must be N x 4 wxyz")
    norms = np.linalg.norm(q, axis=1)
    require(np.isfinite(q).all() and np.all(norms > 1e-12), "Invalid quaternion")
    require(np.all(np.abs(norms - 1) < 1e-3), "Measured quaternion is not unit length")
    return q / norms[:, None]


def quaternion_angular_velocity(quaternions, times):
    """Shortest relative SO(3) increments in the fixed/base frame, in rad/s."""
    q = normalize_quaternions(quaternions)
    times = np.asarray(times, dtype=float)
    require(len(q) == len(times) and np.all(np.diff(times) > 0), "Invalid orientation times")
    # q_next * conjugate(q_previous); q and -q encode the same orientation.
    a, b = q[1:], q[:-1]
    w = np.sum(a * b, axis=1)
    xyz = -a[:, :1] * b[:, 1:] + b[:, :1] * a[:, 1:] - np.cross(a[:, 1:], b[:, 1:])
    xyz[w < 0] *= -1
    w = np.abs(w)
    norms = np.linalg.norm(xyz, axis=1)
    angle = 2 * np.arctan2(norms, np.clip(w, 0, 1))
    factor = np.divide(angle, norms, out=np.full_like(angle, 2.0), where=norms > 1e-12)
    return xyz * factor[:, None] / np.diff(times)[:, None], (times[1:] + times[:-1]) / 2


def trajectory_metrics(arrays, metadata):
    times = np.asarray(arrays["sim_time"], dtype=float)
    require(len(times) > 0, "Empty measured trajectory")
    require(np.array_equal(arrays["physics_index"], np.arange(len(times))), "Missing physics ticks")
    dt = float(metadata["physics_dt"])
    require(dt > 0 and np.allclose(times, np.arange(len(times)) * dt, atol=1e-10, rtol=0),
            "Recorded physics timestamps differ from declared dt")
    indices = arm_indices(metadata["joint_names"])
    q = np.asarray(arrays["q_actual"], dtype=float)
    qd = np.asarray(arrays["qd_actual"], dtype=float)
    state = np.asarray(arrays["state"], dtype=float)
    require(q.shape == qd.shape == (len(times), len(metadata["joint_names"])), "Bad joint array shape")
    require(state.shape == (len(times), 8), "Measured state must be pose8")
    require(np.isfinite(q).all() and np.isfinite(qd).all() and np.isfinite(state).all(),
            "Nonfinite measured state")
    q, qd = q[:, indices], qd[:, indices]
    accel, accel_t = derivative(qd, times)
    jerk, _ = derivative(accel, accel_t)
    speed, speed_t = derivative(state[:, :3], times)
    tcp_accel, tcp_accel_t = derivative(speed, speed_t)
    tcp_jerk, _ = derivative(tcp_accel, tcp_accel_t)
    angular, angular_t = quaternion_angular_velocity(state[:, 3:7], times)
    angular_accel, _ = derivative(angular, angular_t)
    return dict(duration_s=float(times[-1] - times[0]),
                joint_speed_rms=vector_rms(qd),
                joint_acceleration_rms=vector_rms(accel), joint_jerk_rms=vector_rms(jerk),
                tcp_speed_rms=vector_rms(speed), tcp_acceleration_rms=vector_rms(tcp_accel),
                tcp_jerk_rms=vector_rms(tcp_jerk), tcp_angular_speed_rms=vector_rms(angular),
                tcp_angular_acceleration_rms=vector_rms(angular_accel),
                tcp_path_length_m=float(np.linalg.norm(np.diff(state[:, :3], axis=0), axis=1).sum()),
                joint_path_length_rad=float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()))


def distribution(values):
    array = np.asarray([value for value in values if value is not None], dtype=float)
    require(np.isfinite(array).all(), "Nonfinite summary value")
    if not len(array):
        return dict(n=0, mean=None, median=None, p95=None)
    return dict(n=len(array), mean=float(array.mean()), median=float(np.median(array)),
                p95=float(np.percentile(array, 95)))


def aggregate(rows):
    return {metric: distribution(row[metric] for row in rows) for metric in METRIC_UNITS}


def case_key(row):
    return row["floor"], row["offset_x_m"], row["offset_y_m"], row["repeat"]


def validate_coverage(rows, expected_episodes):
    keys = {case_key(row) for row in rows}
    require(len(keys) == len(rows) == expected_episodes, "Duplicate or missing evaluation task")
    if expected_episodes in (12, 120):
        positions = [(0., 0.)] if expected_episodes == 12 else [
            (0., 0.), (-.01, -.025), (-.01, .025), (.01, -.025), (.01, .025)]
        repeats = (1,) if expected_episodes == 12 else (1, 2)
        expected = {(floor, x, y, repeat) for floor in range(24, 36)
                    for x, y in positions for repeat in repeats}
        require(keys == expected, "Fixed evaluation floor/layout/repeat coverage differs")
    return dict(status="pass", episodes=len(keys), all_prescribed_tasks_present=True)


def compare_rows(baseline, candidate):
    left = {case_key(row): row for row in baseline}
    right = {case_key(row): row for row in candidate}
    require(len(left) == len(baseline) and len(right) == len(candidate), "Repeated task pairing key")
    keys = sorted(left.keys() & right.keys())
    common_success = [key for key in keys if left[key]["success"] and right[key]["success"]]
    groups = {}
    for name, selected in (("all_matched", keys), ("common_success", common_success)):
        aa, bb = [left[key] for key in selected], [right[key] for key in selected]
        summary_a, summary_b = aggregate(aa), aggregate(bb)
        metrics = {}
        for metric in METRIC_UNITS:
            usable = [key for key in selected if left[key][metric] is not None and right[key][metric] is not None]
            ratios = [right[key][metric] / left[key][metric] for key in usable if left[key][metric] > 0]
            old, new = summary_a[metric]["median"], summary_b[metric]["median"]
            metrics[metric] = dict(baseline=summary_a[metric], candidate=summary_b[metric],
                median_ratio=None if old in (None, 0) or new is None else new / old,
                per_episode_ratio=distribution(ratios))
        groups[name] = dict(episodes=len(selected), metrics=metrics)
    initial_errors = {}
    unequal_initial = []
    for field in ("q_actual", "qd_actual", "state"):
        errors = [float(np.max(np.abs(np.asarray(left[key]["initial"][field]) -
                                      np.asarray(right[key]["initial"][field])))) for key in keys]
        initial_errors[field] = max(errors, default=None)
    for key in keys:
        if any(not np.allclose(left[key]["initial"][field], right[key]["initial"][field],
                               atol=1e-6, rtol=0) for field in ("q_actual", "qd_actual", "state")):
            unequal_initial.append(list(key))
    return dict(matched_episodes=len(keys), baseline_only=len(left.keys() - right.keys()),
                candidate_only=len(right.keys() - left.keys()),
                gains=sum(not left[key]["success"] and right[key]["success"] for key in keys),
                losses=sum(left[key]["success"] and not right[key]["success"] for key in keys),
                initial_state_max_abs_error=initial_errors, initial_state_mismatched_cases=unequal_initial,
                physical_seed_mismatches=sum(left[key]["physical_seed"] != right[key]["physical_seed"]
                                             for key in keys), **groups)


def independent_slerp(previous, target, alpha):
    """Independent scalar interpolation for auditing, without importing policy code."""
    a, b = normalize_quaternions(np.stack((previous, target)))
    cosine = float(np.dot(a, b))
    if cosine < 0:
        b, cosine = -b, -cosine
    theta = float(np.arccos(np.clip(cosine, 0, 1)))
    if cosine > .9995:
        result = (1 - alpha) * a + alpha * b
    else:
        result = (np.sin((1 - alpha) * theta) * a + np.sin(alpha * theta) * b) / np.sin(theta)
    return result / np.linalg.norm(result)


def audit_action_records(records, expected_mode=None, expected_alpha=None):
    """Check per-slot reset, causality, cross-chunk state and exact EMA outputs."""
    slots, episodes = {}, {}
    errors, norm_errors = [], []
    resets = 0
    chunks = 0
    for record in records:
        env, episode = record["env_id"], record["episode_id"]
        raw = np.asarray(record["raw_actions_pose8"], dtype=float)
        sent = np.asarray(record["sent_actions_pose8"], dtype=float)
        previous = np.asarray(record["previous_sent_pose8"], dtype=float)
        measured = np.asarray(record["observation_state_pose8"], dtype=float)
        require(raw.shape == sent.shape == (7, 8) and previous.shape == measured.shape == (8,),
                "Bad command audit array dimensions")
        require(all(np.isfinite(item).all() for item in (raw, sent, previous, measured)),
                "Nonfinite action audit record")
        mode, alpha = record["mode"], record["alpha"]
        require(mode in ("none", "xyz_ema", "pose_ema") and type(alpha) in (int, float) and 0 < alpha <= 1,
                "Invalid action postprocessing configuration")
        require(record["control_hz"] == 30, "Unexpected control frequency")
        require(expected_mode is None or mode == expected_mode, "Mode differs from manifest")
        require(expected_alpha is None or alpha == expected_alpha, "Alpha differs from manifest")
        reset = env not in slots or slots[env]["episode"] != episode
        require(record["reset"] is reset, "Incorrect per-episode filter reset")
        require(record["chunk_index"] == (0 if reset else slots[env]["chunk_index"] + 1),
                "Noncontiguous command chunks")
        require(np.array_equal(previous, measured if reset else slots[env]["sent"][-1]),
                "Filter initial state or cross-chunk history changed")
        episode_index = record["episode_index"]
        require(episode not in episodes or episodes[episode] == episode_index, "Episode ordinal changed")
        episodes[episode] = episode_index
        expected = raw.copy()
        rolling = previous.copy()
        if mode != "none":
            for step in expected:
                step[:3] = rolling[:3] + alpha * (step[:3] - rolling[:3])
                if mode == "pose_ema":
                    step[3:7] = independent_slerp(rolling[3:7], step[3:7], alpha)
                rolling = step.copy()
        require(np.array_equal(sent[:, 7], raw[:, 7]), "Gripper target modified")
        if mode in ("none", "xyz_ema"):
            require(np.array_equal(sent[:, 3:7], raw[:, 3:7]), "Rotation modified by XYZ-only smoother")
        if mode == "none":
            require(np.array_equal(sent, raw), "Baseline command changed")
        error = float(np.max(np.abs(sent - expected)))
        require(error < 1e-10, "Recorded output differs from independently computed smoothing")
        normalize_quaternions(sent[:, 3:7])
        errors.append(error)
        norm_errors.append(float(np.max(np.abs(np.linalg.norm(sent[:, 3:7], axis=1) - 1))))
        slots[env] = dict(episode=episode, chunk_index=record["chunk_index"], sent=sent)
        resets += int(reset)
        chunks += 1
    require(chunks > 0, "No command records")
    require(len(set(episodes.values())) == len(episodes), "Repeated episode ordinal")
    return dict(status="pass", chunks=chunks, episodes=len(episodes), resets=resets,
                max_independent_smoothing_error=max(errors), max_sent_quaternion_norm_error=max(norm_errors),
                baseline_exact_if_none=True, rotation_exact_if_xyz=True, gripper_exact=True,
                per_episode_reset_and_cross_chunk_history_verified=True,
                note="Command audit covers all targets; actual motion metrics use only executed physics ticks")


def load_rows(run, phase, expected_episodes, allow_incomplete=False):
    mapping = read(run / "run_map.json")
    index = lines(run / "simulation/trajectories/index.jsonl")
    counters = Counter()
    rows = []
    evaluations = {}
    for entry in sorted(index, key=lambda item: item["trajectory_id"]):
        if entry["run_id"] not in mapping:
            continue
        declared = mapping[entry["run_id"]]
        if isinstance(declared, str):
            declared = dict(method=declared, phase="default")
        if declared.get("phase", "default") != phase:
            continue
        require(entry["status"] == "complete", "Incomplete recorded trajectory")
        method = declared["method"]
        directory = Path(entry["directory"])
        metadata = read(directory / "metadata.json")
        require(metadata["run_id"] == entry["run_id"], "Trajectory run identity changed")
        require(metadata["status"] == "complete", "Incomplete trajectory metadata")
        layout = metadata["reset_episode"]
        signature = (method, int(layout["floor"]), float(layout.get("offset_x_m", 0)),
                     float(layout.get("offset_y_m", 0)))
        counters[signature] += 1
        repeat = int(layout.get("repeat", counters[signature]))
        with np.load(directory / "physics.npz", allow_pickle=False) as data:
            metrics = trajectory_metrics(data, metadata)
            initial = {field: data[field][0].tolist() for field in ("q_actual", "qd_actual", "state")}
        info = metadata["info"]
        require(bool(info["success"]) == bool(entry["success"]), "Trajectory outcome mismatch")
        require(abs(metrics["duration_s"] - info["sim_seconds"]) < 1e-9, "Terminal timestamp mismatch")
        rows.append(dict(method=method, phase=phase, run_id=entry["run_id"], trajectory_id=entry["trajectory_id"],
                         directory=str(directory), floor=signature[1], offset_x_m=signature[2],
                         offset_y_m=signature[3], repeat=repeat, physical_seed=metadata["seed"],
                         success=bool(info["success"]), termination=info["termination"],
                         pressed_floors=info.get("pressed_floors", []), samples=metadata["samples"],
                         initial=initial, **metrics))
        path = declared.get("evaluation_directory", str(run / f"{method}_eval"))
        evaluations[method] = Path(path) if Path(path).is_absolute() else run / path
    require(rows, f"No complete trajectories in phase {phase}")
    counts = Counter(row["method"] for row in rows)
    incomplete = {method: count for method, count in counts.items() if count != expected_episodes}
    require(allow_incomplete or not incomplete, f"Episode counts differ from expected {expected_episodes}: {incomplete}")
    rows = [row for row in rows if row["method"] not in incomplete]
    require(rows, "No complete method evaluations yet")
    return rows, evaluations, incomplete


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(run, *, phase="default", baseline="baseline", expected_episodes=120,
            output_dir=None, allow_incomplete=False):
    run = Path(run).resolve()
    rows, evaluations, incomplete = load_rows(run, phase, expected_episodes, allow_incomplete)
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    require(baseline in by_method, f"Missing complete baseline method {baseline}")
    methods, buttons, flat_summary = {}, [], []
    for method, selected in sorted(by_method.items()):
        success = sum(row["success"] for row in selected)
        coverage = validate_coverage(selected, expected_episodes)
        metrics = aggregate(selected)
        matched = compare_rows(by_method[baseline], selected)
        log_path = evaluations[method] / "action_postprocessing.jsonl"
        action_audit = None
        if log_path.is_file():
            config = read(evaluations[method] / "postprocessing.json")
            action_audit = audit_action_records(lines(log_path), config["mode"], config["alpha"])
            require(action_audit["episodes"] == len(selected), "Command/trajectory episode counts differ")
        evaluation_seed = None
        if (evaluations[method] / "summary.json").is_file():
            summary = read(evaluations[method] / "summary.json")
            manifest = read(evaluations[method] / "manifest.json")
            require(summary["state"] == "complete" and summary["episodes"] == len(selected),
                    "Evaluation summary incomplete or count mismatch")
            require(summary["updates"] == summary["session_updates"] == 0, "Evaluation updated parameters")
            require(summary["successes"] == success, "Summary/trajectory successes differ")
            evaluation_seed = manifest["config"]["seed"]
            if (run / "plan.json").is_file():
                plan = read(run / "plan.json")
                require("seed" not in plan or evaluation_seed == plan["seed"], "Evaluation seed differs from plan")
        methods[method] = dict(episodes=len(selected), successes=success, success_rate=success / len(selected),
            failures=len(selected)-success, terminations=dict(Counter(row["termination"] for row in selected)),
            floor_counts=dict(Counter(row["floor"] for row in selected)),
            layout_counts=dict(Counter(f'{row["offset_x_m"]},{row["offset_y_m"]}' for row in selected)),
            repeat_counts=dict(Counter(row["repeat"] for row in selected)),
            all_episodes=metrics, successful_episodes=aggregate([row for row in selected if row["success"]]),
            matched_baseline=matched, action_postprocessing_audit=action_audit,
            coverage=coverage, evaluation_seed=evaluation_seed)
        for floor in sorted({row["floor"] for row in selected}):
            subset = [row for row in selected if row["floor"] == floor]
            wins = sum(row["success"] for row in subset)
            buttons.append(dict(phase=phase, method=method, floor=floor, successes=wins,
                                episodes=len(subset), success_rate=wins / len(subset)))
        flat_summary.append(dict(phase=phase, method=method, episodes=len(selected), successes=success,
            success_rate=success / len(selected), common_success_episodes=matched["common_success"]["episodes"],
            **{f"{metric}_median": metrics[metric]["median"] for metric in METRIC_UNITS},
            **{f"{metric}_common_success_ratio": matched["common_success"]["metrics"][metric]["median_ratio"]
               for metric in METRIC_UNITS}))
    result = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
        source_run=str(run), phase=phase, baseline=baseline, expected_episodes=expected_episodes,
        skipped_incomplete_methods=incomplete, methods=methods,
        definitions=dict(source="Measured PhysX qd_actual and actual gripper_tcp state, not commands",
            derivative="Consecutive divided differences at interval midpoints using saved simulation seconds",
            rms="Square root of mean squared Euclidean vector magnitude over recorded samples",
            arm_joints=list(ARM_JOINT_NAMES), quaternion="Unit wxyz; shortest relative rotation in base frame",
            median_ratio="Candidate median episode RMS / baseline median episode RMS in same task subset",
            pairing="Floor, panel offsets and repeat in reset order; physical reset seed reported separately",
            terminal="Each real episode duration only, no terminal padding or time normalization",
            caution="Acceleration and jerk include reset and contact transients; lower speed can reduce them",
            units=METRIC_UNITS))
    output = Path(output_dir).resolve() if output_dir else run / f"analysis_{phase}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoothing_metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_csv(output / "method_summary.csv", flat_summary)
    write_csv(output / "per_button_success.csv", buttons)
    write_csv(output / "per_episode_smoothness.csv", [
        {key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items() if key != "initial"}
        for row in rows])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--phase", default="default")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--expected-episodes", type=int, default=120)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = analyze(args.run, phase=args.phase, baseline=args.baseline, expected_episodes=args.expected_episodes,
                     output_dir=args.output_dir, allow_incomplete=args.allow_incomplete)
    print(json.dumps({method: {key: value[key] for key in ("episodes", "successes", "success_rate")}
                      for method, value in result["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()
