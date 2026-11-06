"""Six learned actions must survive transport without a learned gripper."""
import hashlib

import numpy as np
import pytest

from pressb.compat.table_buttons_policy import HTTPJoint6Policy, HTTPJointPolicy


def client_and_response(monkeypatch, mutation=None):
    model = dict(native_action_dim=6, state_dim=7, wire_action_dim=7,
                 gripper_learned=False, fixed_gripper_full_width_m=.0002)
    native = np.arange(42, dtype=float).reshape(7, 6) / 10
    noise = np.arange(42, dtype=np.float32).reshape(7, 6) / 42
    actions = np.column_stack((native, np.full(7, .02)))
    evidence = dict(native_actions=native.tolist(), native_normalized_actions=(native/180).tolist(),
                    initial_noise=noise.tolist())
    trace = dict(normalized_actions=(actions/np.array([180]*6+[8])).tolist(),
                 initial_noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest())
    if mutation:
        mutation(actions, evidence, trace)

    def predict(self, **_):
        self.native_evidence = evidence
        return actions, trace

    monkeypatch.setattr(HTTPJointPolicy, 'predict', predict)
    policy = HTTPJoint6Policy('http://localhost', 'medium_red_joint6', model_contract=model)
    policy.model = model
    return policy


def test_preserve_native_outputs_and_actual_six_dimensional_noise(monkeypatch):
    actions, trace = client_and_response(monkeypatch).predict()
    assert np.asarray(trace['initial_noise']).shape == (7, 6)
    np.testing.assert_array_equal(actions[:, :6], trace['native_actions'])
    np.testing.assert_allclose(actions[:, 6], .02)


@pytest.mark.parametrize('mutation,match', [
    (lambda a, e, t: a.__setitem__((0, 6), .03), 'fixed width'),
    (lambda a, e, t: e['native_actions'][0].__setitem__(0, 42), 'native six-action'),
    (lambda a, e, t: e['initial_noise'][0].__setitem__(0, 42), 'digest'),
    (lambda a, e, t: e.__setitem__('initial_noise', [[0]*7]*7), 'actions/noise'),
])
def test_reject_inconsistent_joint6_evidence(monkeypatch, mutation, match):
    with pytest.raises(RuntimeError, match=match):
        client_and_response(monkeypatch, mutation).predict()
