"""Scripted contact feedback follows plan timing, including color detours."""
import numpy as np
import pytest

from pressb.table_buttons.execution import (
    contact_limited_command, in_press_window, press_phase_times,
)


def test_historical_plan_executes_original_hold_and_release():
    times = press_phase_times({})
    assert times == dict(press_start=3.1, release_start=4.45, return_hover_end=5.55)
    path_times = np.arange(235)/30
    path = np.column_stack([np.sin(path_times+i) for i in range(7)])
    held = np.linspace(.1, .7, 7)
    for seconds in np.arange(937)/120:
        nominal = np.array([np.interp(seconds, path_times, path[:, j]) for j in range(7)])
        expected = nominal
        if 3.1 <= seconds < 4.45:
            expected = held
        elif 4.45 <= seconds < 5.55:
            u = (seconds-4.45)/1.1
            blend = u*u*u*(10+u*(-15+6*u))
            hover = np.array([np.interp(5.55, path_times, path[:, j]) for j in range(7)])
            expected = held+(hover-held)*blend
        actual = contact_limited_command(seconds, nominal, held, path_times, path, times)
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
        assert in_press_window(seconds, times) == (3.1 <= seconds < 4.45)


def test_delayed_contact_holds_through_new_press_window_then_retreats():
    phases = press_phase_times({'phase_times': dict(
        press_start=5.1, release_start=6.45, return_hover_end=8.65)})
    held = np.full(7, .2)
    nominal = np.full(7, -.7)
    path_times = np.array([0., 5.55, 8.65, 10.])
    path = np.array([np.zeros(7), np.ones(7), np.full(7, .8), np.zeros(7)])
    assert not in_press_window(4., phases)
    assert in_press_window(5.1, phases)
    assert in_press_window(6.449, phases)
    assert not in_press_window(6.45, phases)
    np.testing.assert_array_equal(contact_limited_command(
        4., nominal, held, path_times, path, phases), nominal)
    np.testing.assert_array_equal(contact_limited_command(
        6., nominal, held, path_times, path, phases), held)
    np.testing.assert_allclose(contact_limited_command(
        7.55, nominal, held, path_times, path, phases), np.full(7, .5))
    np.testing.assert_array_equal(contact_limited_command(
        8.65, nominal, held, path_times, path, phases), nominal)


def test_without_contact_nominal_target_is_unchanged():
    nominal = np.arange(7, dtype=float)
    for seconds in (0., 3.5, 5., 7.):
        assert contact_limited_command(seconds, nominal, None, None, None, press_phase_times({})) is nominal


@pytest.mark.parametrize('values', [
    dict(press_start=3.1, release_start=3.1, return_hover_end=5.55),
    dict(press_start=3.1, release_start=5.55, return_hover_end=4.45),
    dict(press_start=-1., release_start=4.45, return_hover_end=5.55),
    dict(press_start=3.1, release_start=np.nan, return_hover_end=5.55),
])
def test_invalid_feedback_window_fails_before_execution(values):
    with pytest.raises(ValueError):
        press_phase_times({'phase_times': values})
