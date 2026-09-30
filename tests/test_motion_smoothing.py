"""Causal timing, isolation and physical-bound checks for command smoothing."""
import json
from pathlib import Path

import numpy as np
import pytest

from pressb.kinematics import PiperKinematics
from pressb.motion_smoothing import JointCommandSmoother, smoothing_settings, validate_smoothing_window


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("window", [1, 3, 5, 7, 9, 11, np.int64(5)])
def test_supported_windows_and_nominal_delay(window):
    settings = smoothing_settings(window)
    assert validate_smoothing_window(window) == window
    assert settings == dict(kind="linear_joint_interpolation_then_causal_mean", window=int(window),
                           physics_hz=120, nominal_delay_s=(window - 1) / 240,
                           initial_history="repeat_initial_command", reset="per_episode", history="cross_chunk")
    json.dumps(settings, allow_nan=False)


@pytest.mark.parametrize("window", [0, -1, 2, 4, 6, 12, 13, 5., "5", True, np.bool_(False), None, float("nan")])
def test_invalid_windows_rejected(window):
    with pytest.raises(ValueError, match="Smoothing window"):
        JointCommandSmoother(np.zeros(6), window)
    with pytest.raises(ValueError, match="Smoothing window"):
        smoothing_settings(window)


@pytest.mark.parametrize("hz", [0, -120, 120., True, None, float("inf")])
def test_invalid_physics_rate_rejected(hz):
    with pytest.raises(ValueError, match="physics_hz"):
        smoothing_settings(5, hz)


def test_step_response_converges_and_impulse_delay_is_two_physics_ticks():
    smoother = JointCommandSmoother(np.zeros(6), 5)
    response = np.array([smoother.step(np.ones(6)) for _ in range(8)])
    np.testing.assert_allclose(response[:, 0], [.2, .4, .6, .8, 1., 1., 1., 1.], rtol=0, atol=1e-15)
    impulse = JointCommandSmoother(np.zeros(6), 5)
    values = np.array([impulse.step(np.ones(6) if tick == 0 else np.zeros(6))[0] for tick in range(9)])
    np.testing.assert_allclose(values, [.2]*5 + [0.]*4, rtol=0, atol=1e-15)
    centroid_s = float(np.dot(np.arange(len(values)) / 120, values) / values.sum())
    assert centroid_s == pytest.approx(2 / 120, abs=1e-15)
    assert centroid_s == smoothing_settings(5)["nominal_delay_s"]


def test_linear_ramp_has_documented_delay_after_history_fills():
    initial = np.linspace(-.3, .4, 6)
    velocity_per_tick = np.linspace(.001, .003, 6)
    smoother = JointCommandSmoother(initial, 5)
    for tick in range(1, 30):
        actual = smoother.step(initial + tick * velocity_per_tick)
        if tick >= 5:
            np.testing.assert_allclose(actual, initial + (tick - 2) * velocity_per_tick, atol=1e-15, rtol=0)


def test_history_continues_across_chunks_and_environments_are_isolated():
    initial = np.zeros(6)
    stream = np.arange(1, 85, dtype=float)[:, None] * np.linspace(.001, .002, 6)
    uninterrupted = JointCommandSmoother(initial, 5)
    expected = np.array([uninterrupted.step(value) for value in stream])
    first_env = JointCommandSmoother(initial, 5)
    other_env = JointCommandSmoother(np.ones(6), 5)
    actual = []
    # Seven 30 Hz actions each contain four 120 Hz interpolation commands.
    for chunk in np.split(stream, 3):
        for command in chunk:
            actual.append(first_env.step(command))
            np.testing.assert_array_equal(other_env.step(np.ones(6)), np.ones(6))
    np.testing.assert_array_equal(actual, expected)
    next_episode = JointCommandSmoother(initial, 5)
    np.testing.assert_allclose(next_episode.step(stream[0]), stream[0] / 5, atol=1e-16, rtol=0)
    assert not np.array_equal(actual[28], JointCommandSmoother(initial, 5).step(stream[28]))


def test_input_and_returned_arrays_never_alias_history():
    initial = np.zeros(6)
    value = np.ones(6, dtype=np.float32)
    smoother = JointCommandSmoother(initial, 5)
    initial[:] = 50
    returned = smoother.step(value)
    assert returned.dtype == np.float64
    np.testing.assert_array_equal(value, np.ones(6))
    value[:] = 100
    returned[:] = -100
    np.testing.assert_allclose(smoother.step(np.ones(6)), np.full(6, .4), rtol=0, atol=1e-15)


@pytest.mark.parametrize("bad", [np.zeros(5), np.zeros((1, 6)), [0., 0., 0., 0., 0., float("nan")],
                                [float("inf")]*6, [1j]*6, [None]*6, "bad"])
def test_invalid_commands_do_not_mutate_existing_history(bad):
    smoother = JointCommandSmoother(np.zeros(6), 5)
    with pytest.raises(ValueError, match="six finite real"):
        smoother.step(bad)
    np.testing.assert_allclose(smoother.step(np.ones(6)), np.full(6, .2), rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="six finite real"):
        JointCommandSmoother(bad, 5)


def test_window_one_preserves_float64_commands_exactly():
    smoother = JointCommandSmoother(np.zeros(6), 1)
    values = [np.array([0., -0., np.pi, np.nextafter(1., 2.), -.7, .3]),
              np.arange(6, dtype=np.float64)[::-1], np.linspace(-1, 1, 6, dtype=np.float32)]
    for value in values:
        result = smoother.step(value)
        assert result.dtype == np.float64
        assert result.tobytes() == np.asarray(value, dtype=np.float64).tobytes()
        assert not np.shares_memory(result, value)


def test_default_filter_has_one_tick_ramp_delay():
    smoother = JointCommandSmoother(np.zeros(6))
    assert smoother.window == 3
    response = np.array([smoother.step(np.full(6, tick * .001)) for tick in range(1, 9)])
    np.testing.assert_allclose(response[2:, 0], np.arange(2, 8) * .001, atol=1e-16, rtol=0)
    assert smoothing_settings(smoother.window)["nominal_delay_s"] == 1 / 120


@pytest.mark.parametrize("window", [1, 3, 5, 11])
def test_convex_filter_preserves_actual_urdf_limits_and_velocity_bound(window):
    cfg = json.loads((ROOT / "configs/scene.json").read_text())
    kin = PiperKinematics(ROOT / cfg["robot_urdf"], tip_offset=cfg["tip_offset"])
    initial = np.where(np.arange(6) % 2, kin.upper, kin.lower)
    direction = np.where(np.arange(6) % 2, -1., 1.)
    per_tick = kin.velocity_limits / 120
    smoother = JointCommandSmoother(initial, window)
    commands, filtered = [initial.copy()], [initial.copy()]
    for _ in range(900):
        proposed = commands[-1] + direction * per_tick
        next_command = np.clip(proposed, kin.lower, kin.upper)
        direction = np.where((proposed >= kin.upper) | (proposed <= kin.lower), -direction, direction)
        commands.append(next_command)
        filtered.append(smoother.step(next_command))
    commands, filtered = np.array(commands), np.array(filtered)
    assert np.all(filtered >= kin.lower - 1e-14) and np.all(filtered <= kin.upper + 1e-14)
    assert np.all(np.abs(np.diff(commands, axis=0)) <= per_tick + 1e-13)
    assert np.all(np.abs(np.diff(filtered, axis=0)) <= per_tick + 1e-13)


def test_five_point_average_reduces_jerk_of_piecewise_linear_action_execution():
    # A changing 30 Hz action sequence produces velocity jumps at four-tick
    # boundaries after linear interpolation. Compare the same 120 Hz timeline.
    targets = .04 * np.sin(np.arange(61) * 1.3)
    commands = [np.repeat(targets[0], 6)]
    for start, end in zip(targets[:-1], targets[1:]):
        commands.extend(np.repeat(start + fraction * (end - start), 6) for fraction in (.25, .5, .75, 1.))
    commands = np.asarray(commands)
    smoother = JointCommandSmoother(commands[0], 5)
    filtered = np.vstack([commands[0], [smoother.step(row) for row in commands[1:]]])
    raw_jerk = np.diff(commands, n=3, axis=0) * 120**3
    filtered_jerk = np.diff(filtered, n=3, axis=0) * 120**3
    assert filtered.shape == commands.shape
    assert np.sqrt(np.mean(filtered_jerk**2)) < .5 * np.sqrt(np.mean(raw_jerk**2))
    assert np.max(np.abs(filtered_jerk)) < np.max(np.abs(raw_jerk))
