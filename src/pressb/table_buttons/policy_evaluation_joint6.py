"""Preserve native six-action evidence through the existing seven-channel RPC."""
from copy import deepcopy
import hashlib

import numpy as np

from .policy_evaluation import HTTPJointPolicy


class HTTPJoint6Policy(HTTPJointPolicy):
    def __init__(self, *args, model_contract, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_contract = deepcopy(model_contract)
        self.native_evidence = None

    def load(self):
        model = super().load()
        for key, value in self.model_contract.items():
            if model.get(key) != value:
                raise RuntimeError(f"Joint6 model differs from the frozen contract: {key}")
        if (model.get('native_action_dim') != 6 or model.get('state_dim') != 7
                or model.get('wire_action_dim') != 7 or model.get('gripper_learned') is not False):
            raise RuntimeError('Expected six learned arm actions and a fixed transport gripper')
        return model

    def call(self, route, body=None):
        result = super().call(route, body)
        if route == '/predict':
            self.native_evidence = None
            predictions = result.get('predictions', [])
            if len(predictions) == 1:
                self.native_evidence = {key: deepcopy(predictions[0][key]) for key in
                    ('native_actions', 'native_normalized_actions', 'initial_noise') if key in predictions[0]}
        return result

    def predict(self, **request):
        self.native_evidence = None
        actions, trace = super().predict(**request)
        evidence = self.native_evidence or {}
        arrays = {key: np.asarray(evidence.get(key), dtype=np.float64) for key in
                  ('native_actions', 'native_normalized_actions', 'initial_noise')}
        if any(value.shape != (7, 6) or not np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError('Joint6 response lacks finite seven-step native actions/noise')
        if (not np.allclose(actions[:, :6], arrays['native_actions'], atol=2e-5, rtol=2e-6)
                or not np.allclose(np.asarray(trace['normalized_actions'])[:, :6],
                                   arrays['native_normalized_actions'], atol=2e-7, rtol=2e-6)):
            raise RuntimeError('Wire actions differ from native six-action output')
        width_cm = self.model['fixed_gripper_full_width_m'] * 100.
        if not np.allclose(actions[:, 6], width_cm, atol=1e-8, rtol=0.):
            raise RuntimeError('Joint6 transport gripper is not the declared fixed width')
        noise = np.asarray(arrays['initial_noise'], dtype=np.float32)
        if hashlib.sha256(noise.tobytes()).hexdigest() != trace.get('initial_noise_sha256'):
            raise RuntimeError('Native six-dimensional noise differs from its recorded digest')
        trace.update(evidence)
        return actions, trace
