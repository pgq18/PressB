"""Independent smoothing evidence must reject wrong timing and endpoint claims."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_policy_eval import (command_smoothing_evidence, executed_action_evidence,
                               request_seed_evidence, smoothing_settings)
from diagnose_policy_eval import derivative_metrics, motion_smoothness


def metadata(window):
    return dict(motion_smoothing=dict(kind='linear_joint_interpolation_then_causal_mean', window=window,
        physics_hz=120, nominal_delay_s=(window - 1) / 240, initial_history='repeat_initial_command',
        reset='per_episode', history='cross_chunk'))


@pytest.mark.parametrize('window', [1, 3, 5, 7, 9, 11])
def test_ramp_matches_analytic_delay_with_repeat_initial_history(window):
    count = 40
    raw = np.repeat((.2 + np.arange(count) / 100)[:, None], 6, axis=1)
    # Closed-form mean of a ramp padded with the initial command.
    offset = np.array([i * (i + 1) / (2 * window) if i < window else i - (window - 1) / 2
                       for i in range(count)]) / 100
    sent = np.repeat((.2 + offset)[:, None], 6, axis=1).astype(np.float32)
    restored, settings, error = command_smoothing_evidence(metadata(window),
        dict(q_command=sent, q_command_unsmoothed=raw))
    np.testing.assert_array_equal(restored, raw)
    assert settings['window'] == window
    assert error <= np.spacing(np.float32(.6))


def test_impulse_history_must_cross_seven_action_chunk_boundary():
    raw = np.zeros((40, 6))
    raw[27] = .1  # A seven-action/28-physics-step boundary follows.
    sent = np.zeros_like(raw, dtype=np.float32)
    sent[27:32] = .02
    command_smoothing_evidence(metadata(5), dict(q_command=sent, q_command_unsmoothed=raw))
    reset_at_chunk = sent.copy()
    reset_at_chunk[28:32] = 0
    with pytest.raises(ValueError, match='causal moving mean'):
        command_smoothing_evidence(metadata(5), dict(q_command=reset_at_chunk, q_command_unsmoothed=raw))
    future_looking = np.roll(sent, -2, axis=0)
    with pytest.raises(ValueError, match='causal moving mean'):
        command_smoothing_evidence(metadata(5), dict(q_command=future_looking, q_command_unsmoothed=raw))


def test_legacy_window_one_and_new_evidence_presence_rules():
    sent = np.arange(24, dtype=np.float32).reshape(4, 6) / 100
    raw, settings, error = command_smoothing_evidence({}, dict(q_command=sent))
    np.testing.assert_array_equal(raw, sent)
    assert settings['window'] == 1 and error == 0
    with pytest.raises(ValueError, match='recorded together'):
        command_smoothing_evidence(metadata(1), dict(q_command=sent))
    with pytest.raises(ValueError, match='recorded together'):
        command_smoothing_evidence({}, dict(q_command=sent, q_command_unsmoothed=sent))


@pytest.mark.parametrize('change', ['window', 'delay', 'history', 'initial'])
def test_filter_contract_is_explicit(change):
    record = metadata(5)
    key, value = dict(window=('window', 4), delay=('nominal_delay_s', 0),
                     history=('history', 'per_chunk'), initial=('initial_history', 'zeros'))[change]
    record['motion_smoothing'][key] = value
    with pytest.raises(ValueError):
        smoothing_settings(record)


class TranslationFK:
    def batch(self, joints):
        result = np.repeat(np.eye(4)[None], len(joints), axis=0)
        result[:, :3, 3] = joints[:, :3]
        return result


def test_executed_residual_uses_filtered_partial_endpoint_not_full_ik_target():
    sent = np.zeros((3, 6), dtype=np.float32)
    sent[1, 0], sent[2, 0] = .125, .25
    pose = np.array([1., 0., 0., 1., 0., 0., 0., .008])
    action = dict(physics_start_index=0, physics_end_index=2, q_target=[1., 0., 0., 0., 0., 0.],
        q_executed_before=sent[0].tolist(), q_executed_endpoint=sent[2].tolist(),
        executed_position_residual_m=.75, executed_rotation_residual_rad=0.)
    assert executed_action_evidence(action, pose, sent, TranslationFK()) == (.75, 0.)
    action['q_executed_endpoint'] = action['q_target']
    with pytest.raises(ValueError, match='sent command'):
        executed_action_evidence(action, pose, sent, TranslationFK())
    action['q_executed_endpoint'] = sent[2].tolist()
    action['executed_position_residual_m'] = 0.
    with pytest.raises(ValueError, match='position residual'):
        executed_action_evidence(action, pose, sent, TranslationFK())


def test_derivative_units_scaling_and_no_terminal_padding():
    t = np.arange(10) / 120
    joint = np.repeat((t ** 3)[:, None], 6, axis=1)
    metrics = derivative_metrics(joint, 120, 'rad')
    assert metrics['velocity']['samples'] == 9
    assert metrics['acceleration']['samples'] == 8
    assert metrics['jerk']['samples'] == 7
    assert metrics['jerk']['unit'] == 'rad/s^3'
    for key in ('mean_abs', 'rms', 'peak_abs'):
        assert metrics['jerk'][key] == pytest.approx(6)
    state = np.zeros((10, 8))
    state[:, :3] = np.repeat((2 * t ** 3)[:, None], 3, axis=1)
    all_metrics = motion_smoothness(dict(state=state, q_command=joint))
    assert all_metrics['actual_tcp_position']['jerk']['rms'] == pytest.approx(12)
    assert all_metrics['actual_tcp_position']['jerk']['unit'] == 'm/s^3'
    assert all_metrics['unsmoothed_joint_command'] == all_metrics['executed_joint_command']


def test_short_terminal_record_has_no_invented_jerk_samples():
    metric = derivative_metrics(np.zeros((2, 3)), 120, 'm')['jerk']
    assert metric == dict(unit='m/s^3', samples=0, mean_abs=None, rms=None, peak_abs=None)


@pytest.mark.parametrize('floor', [24, 33, 35])
def test_subset_requests_retain_the_full_floor_schedule_seed(floor):
    meta = dict(seed_episode_index=floor - 24, repeat=0, floor=floor)
    requests = [dict(seed=99 + (floor - 24) * 10000 + chunk, chunk_index=chunk) for chunk in range(3)]
    assert request_seed_evidence(meta, requests, 99) == 99
    requests[1]['seed'] += 10000
    with pytest.raises(ValueError, match='inference seed'):
        request_seed_evidence(meta, requests, 99)
