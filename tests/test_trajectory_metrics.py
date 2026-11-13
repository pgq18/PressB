from copy import deepcopy
import json

import numpy as np
import pytest

from pressb.runtime.trajectory_metrics import measured_motion_metrics, plan_motion_metrics


def test_small_motion_is_distinguished_from_exact_duplicates_and_threshold_boundary():
    # Four intervals: repeat, slow movement, threshold equality, gripper motion.
    states = np.zeros((5, 7))
    states[2:, 0] = np.deg2rad(.025)
    states[3, 1] = np.deg2rad(.05)
    states[4] = states[3]
    states[4, 6] = .0001
    before = states.copy()
    metrics = measured_motion_metrics(states, 30)
    assert metrics['sample_count'] == 5
    assert metrics['interval_count'] == metrics['comparison_count'] == 4
    assert metrics['duration_s'] == 4/30
    assert metrics['exact_duplicate_count'] == 1
    assert metrics['near_static_interval_count'] == 2
    assert metrics['slow_nonzero_interval_count'] == 1
    assert metrics['exact_duplicate_fraction'] == .25
    assert metrics['near_static_interval_fraction'] == .5
    assert metrics['max_gripper_width_m'] == .0001
    assert metrics['near_static_thresholds']['joint_rate_deg_s'] == 1.5
    np.testing.assert_array_equal(states, before)
    json.dumps(metrics, allow_nan=False)


def test_plan_timeline_holds_and_initial_comparison_do_not_drop_or_retime_targets():
    plan = dict(actions=[[0]*7, [0]*7, [0]*6+[.1]], fps=30,
                phases=['approach', 'approach', 'release'],
                waypoints=[dict(phase='approach', hold_seconds=1/30),
                           dict(phase='release', hold_seconds=0)])
    original = deepcopy(plan)
    metrics = plan_motion_metrics(plan)
    assert metrics['action_count'] == 3
    assert metrics['expected_recorded_frame_count'] == 4
    assert metrics['duration_s'] == 3/30
    assert metrics['comparison_count'] == 2
    assert metrics['exact_duplicate_count'] == 1
    assert metrics['phase_frames'] == {'approach': 2, 'release': 1}
    assert metrics['planned_hold_frames'] == 1
    assert metrics['planned_hold_frames_by_phase'] == {'approach': 1, 'release': 0}
    assert metrics['max_gripper_width_m'] == .1
    assert not metrics['row_filtering'] and not metrics['retimed']
    initial_metrics = plan_motion_metrics(plan, [0]*7)
    assert initial_metrics['comparison_count'] == 3
    assert initial_metrics['exact_duplicate_count'] == 2
    assert initial_metrics['comparison_reference'] == 'initial_state_then_targets'
    assert plan == original
    json.dumps(initial_metrics, allow_nan=False)


def test_single_boundary_state_has_no_elapsed_time_or_motion_comparisons():
    metrics = measured_motion_metrics([[0]*7], 30)
    assert metrics['duration_s'] == 0
    assert metrics['comparison_count'] == 0
    assert metrics['near_static_interval_fraction'] == 0
    json.dumps(metrics, allow_nan=False)
    plan = plan_motion_metrics(dict(actions=[[0]*7], fps=30))
    assert plan['duration_s'] == 1/30
    assert plan['comparison_count'] == 0


@pytest.mark.parametrize('values', [[], [[0]*6], [[float('nan')]*7], [0]*7])
def test_invalid_shape_or_nonfinite_series_is_rejected(values):
    with pytest.raises(ValueError):
        measured_motion_metrics(values, 30)
    with pytest.raises(ValueError):
        plan_motion_metrics(dict(actions=values, fps=30))


@pytest.mark.parametrize('fps', [0, -1, float('nan'), float('inf'), None])
def test_invalid_control_rate_is_rejected(fps):
    with pytest.raises(ValueError):
        measured_motion_metrics([[0]*7], fps)


def test_invalid_plan_phase_length_hold_and_initial_state_are_rejected():
    plan = dict(actions=[[0]*7], fps=30)
    with pytest.raises(ValueError, match='phases'):
        plan_motion_metrics({**plan, 'phases': []})
    with pytest.raises(ValueError, match='hold_seconds'):
        plan_motion_metrics({**plan, 'waypoints': [{'hold_seconds': -1}]})
    with pytest.raises(ValueError, match='initial_state'):
        plan_motion_metrics(plan, [0]*6)
