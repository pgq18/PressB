"""Measured motion diagnostics: derivative units, data integrity and paired tasks."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_rl_smoothing.py"
SPEC = importlib.util.spec_from_file_location("analyze_rl_smoothing", SCRIPT)
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def measured(count=241, dt=1 / 120):
    time = np.arange(count) * dt
    state = np.zeros((count, 8))
    state[:, 3] = 1
    return dict(sim_time=time, physics_index=np.arange(count),
                q_actual=np.zeros((count, 8)), qd_actual=np.zeros((count, 8)), state=state), {
                    "physics_dt": dt, "joint_names": [f"joint{i}" for i in range(1, 9)]}


def test_derivative_uses_real_seconds_and_midpoints():
    time = np.array([0., .01, .03, .1])
    speed, middle = metrics.derivative((3 * time)[:, None], time)
    np.testing.assert_allclose(speed, 3.)
    np.testing.assert_allclose(middle, [.005, .02, .065])


def test_quadratic_joint_velocity_has_correct_jerk_units():
    arrays, meta = measured()
    time = arrays["sim_time"]
    arrays["qd_actual"][:, 0] = 2 * time * time
    result = metrics.trajectory_metrics(arrays, meta)
    assert result["joint_jerk_rms"] == pytest.approx(4.)
    middle = (time[1:] + time[:-1]) / 2
    assert result["joint_acceleration_rms"] == pytest.approx(np.sqrt(np.mean((4 * middle) ** 2)))


def test_cubic_tcp_position_has_correct_third_derivative():
    arrays, meta = measured()
    time = arrays["sim_time"]
    arrays["state"][:, 0] = .5 * time ** 3
    result = metrics.trajectory_metrics(arrays, meta)
    assert result["tcp_jerk_rms"] == pytest.approx(3.)
    assert result["tcp_path_length_m"] == pytest.approx(4.)


def test_arm_joint_selection_uses_names_and_excludes_gripper():
    arrays, meta = measured()
    meta["joint_names"] = ["joint8", "joint6", "joint5", "joint4", "joint3", "joint2", "joint1", "joint7"]
    time = arrays["sim_time"]
    arrays["qd_actual"][:, 0] = 10000 * time ** 2
    arrays["qd_actual"][:, -1] = -10000 * time ** 2
    arrays["qd_actual"][:, 6] = 2 * time ** 2
    assert metrics.trajectory_metrics(arrays, meta)["joint_jerk_rms"] == pytest.approx(4.)


@pytest.mark.parametrize("names", [["joint1"] * 8, [f"joint{i}" for i in range(2, 10)]])
def test_unknown_or_duplicate_joints_rejected(names):
    with pytest.raises(ValueError):
        metrics.arm_indices(names)


def test_quaternion_sign_flips_do_not_create_angular_jitter():
    arrays, meta = measured()
    time = arrays["sim_time"]
    angle = .6 * time
    arrays["state"][:, 3] = np.cos(angle / 2)
    arrays["state"][:, 6] = np.sin(angle / 2)
    arrays["state"][::2, 3:7] *= -1
    result = metrics.trajectory_metrics(arrays, meta)
    assert result["tcp_angular_speed_rms"] == pytest.approx(.6)
    assert result["tcp_angular_acceleration_rms"] < 1e-10


@pytest.mark.parametrize("bad", [[0, 0, 0, 0], [2, 0, 0, 0], [float("nan"), 0, 0, 0]])
def test_invalid_measured_quaternions_rejected(bad):
    with pytest.raises(ValueError):
        metrics.normalize_quaternions([bad])


@pytest.mark.parametrize("count", [1, 2, 3])
def test_short_trajectory_unavailable_derivatives_are_null(count):
    arrays, meta = measured(count)
    result = metrics.trajectory_metrics(arrays, meta)
    assert result["duration_s"] == pytest.approx((count - 1) / 120)
    assert result["tcp_jerk_rms"] is None
    assert result["joint_jerk_rms"] == (0.0 if count == 3 else None)


def test_duplicate_or_missing_timestamps_rejected():
    with pytest.raises(ValueError):
        metrics.derivative(np.zeros((3, 2)), [0., 0., .1])
    arrays, meta = measured()
    arrays["physics_index"][5] = 6
    with pytest.raises(ValueError, match="Missing physics ticks"):
        metrics.trajectory_metrics(arrays, meta)


def row(floor, success, value, repeat=1):
    return dict(floor=floor, offset_x_m=0., offset_y_m=0., repeat=repeat,
                physical_seed=1, success=success,
                initial={field: [0.] for field in ("q_actual", "qd_actual", "state")},
                **{key: value for key in metrics.METRIC_UNITS})


def test_common_success_pairing_does_not_select_failed_short_episodes():
    baseline = [row(24, True, 100), row(25, True, 5), row(26, False, 10)]
    candidate = [row(26, True, 20), row(24, False, 1), row(25, True, 4)]
    result = metrics.compare_rows(baseline, candidate)
    assert result["matched_episodes"] == 3
    assert result["gains"] == result["losses"] == 1
    assert result["common_success"]["episodes"] == 1
    assert result["common_success"]["metrics"]["joint_jerk_rms"]["median_ratio"] == .8


def test_repeat_part_of_pair_key_but_physical_seed_reported_separately():
    left = [row(24, True, 2, 1), row(24, True, 10, 2)]
    right = [row(24, True, 5, 2), row(24, True, 1, 1)]
    right[0]["physical_seed"] = 7
    right[1]["initial"]["state"] = [.1]
    result = metrics.compare_rows(left, right)
    assert result["matched_episodes"] == 2
    assert result["physical_seed_mismatches"] == 1
    assert result["initial_state_max_abs_error"]["state"] == .1
    assert len(result["initial_state_mismatched_cases"]) == 1
    assert result["common_success"]["metrics"]["joint_jerk_rms"]["per_episode_ratio"]["median"] == .5


def test_fixed_120_coverage_rejects_wrong_layout_despite_right_count():
    positions = [(0., 0.), (-.01, -.025), (-.01, .025), (.01, -.025), (.01, .025)]
    rows = [dict(row(floor, True, 0, repeat), offset_x_m=x, offset_y_m=y)
            for floor in range(24, 36) for x, y in positions for repeat in (1, 2)]
    assert metrics.validate_coverage(rows, 120)["status"] == "pass"
    rows[0]["offset_x_m"] = .003
    with pytest.raises(ValueError, match="coverage"):
        metrics.validate_coverage(rows, 120)


def action_record():
    raw = np.zeros((7, 8))
    raw[:, 0] = 1
    raw[:, 3] = 1
    raw[:, 7] = .008
    previous = raw[0].copy()
    previous[0] = 0
    sent = raw.copy()
    sent[:, 0] = 1 - .5 ** np.arange(1, 8)
    return dict(env_id=0, episode_id="episode0", episode_index=0, chunk_index=0, reset=True,
                mode="xyz_ema", alpha=.5, control_hz=30, observation_state_pose8=previous.tolist(),
                previous_sent_pose8=previous.tolist(), raw_actions_pose8=raw.tolist(),
                sent_actions_pose8=sent.tolist())


def test_independent_action_audit_accepts_analytic_causal_ema():
    result = metrics.audit_action_records([action_record()], "xyz_ema", .5)
    assert result["status"] == "pass"
    assert result["max_independent_smoothing_error"] == 0


@pytest.mark.parametrize("change", ["output", "rotation", "reset", "chunk", "initial"])
def test_independent_action_audit_rejects_corruption(change):
    record = action_record()
    if change == "output":
        record["sent_actions_pose8"][3][0] += .1
    elif change == "rotation":
        record["sent_actions_pose8"][3][4] += .1
    elif change == "reset":
        record["reset"] = False
    elif change == "chunk":
        record["chunk_index"] = 1
    else:
        record["previous_sent_pose8"][0] = .1
    with pytest.raises(ValueError):
        metrics.audit_action_records([record])
