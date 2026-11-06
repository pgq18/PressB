"""Legacy JSON HTTP joint7/joint6 clients retained for archived evaluations.

New model integrations use pressb.models and the common runtime contracts.
This transport preserves the original units, normalization and response checks.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
from PIL import Image

from pressb.tasks.table_buttons.policy_contract import InvalidPolicyAction, SCALE, joint7


def png_base64(rgb):
    rgb = np.asarray(rgb)
    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError("Policy cameras must provide RGB uint8 HWC 480x640")
    stream = BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG", compress_level=1)
    return base64.b64encode(stream.getvalue()).decode("ascii")


class HTTPJointPolicy:
    """Synchronous JSON RPC; callers do not advance simulation while waiting."""
    def __init__(self, endpoint, model_id, *, timeout=180.):
        if not str(endpoint).startswith(("http://", "https://")):
            raise ValueError("Expected an HTTP(S) inference endpoint")
        self.endpoint = str(endpoint).rstrip("/")
        self.model_id, self.timeout = str(model_id), float(timeout)
        self.model = None

    def call(self, route, body=None):
        payload = None if body is None else json.dumps(body, allow_nan=False).encode()
        request = Request(self.endpoint + route, data=payload,
                          headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            if error.code == 422:
                details = json.loads(error.read())
                if details.get('error_kind') == 'invalid_policy_action':
                    raise InvalidPolicyAction(details.get('error', 'Server returned invalid model actions')) from error
            raise
        if not isinstance(result, dict):
            raise RuntimeError("Inference server did not return a JSON object")
        return result

    def load(self):
        body = self.call("/load", dict(model_id=self.model_id))
        model = body.get("model", body)
        if model.get("model_id") != self.model_id or not model.get("model_sha256"):
            raise RuntimeError("Loaded model identity/hash missing or mismatched")
        self.model = deepcopy(model)
        return deepcopy(self.model)

    def reset(self, episode_id):
        return self.call("/reset", dict(model_id=self.model_id, episode_id=episode_id))

    def predict(self, *, episode_id, request_id, instruction, state, images, noise_seed):
        sample = dict(episode_id=episode_id, request_id=request_id, instruction=instruction,
                      state=joint7(state).tolist(), images={name: png_base64(images[name]) for name in ("up", "wrist")},
                      noise_seed=int(noise_seed))
        started = time.perf_counter()
        body = self.call("/predict", dict(model_id=self.model_id, samples=[sample]))
        elapsed = time.perf_counter() - started
        model = body.get("model", {})
        if self.model is None or model.get("model_id") != self.model_id or model.get("model_sha256") != self.model["model_sha256"]:
            raise RuntimeError("Inference model changed during evaluation")
        predictions = body.get("predictions", [])
        if len(predictions) != 1:
            raise RuntimeError("Expected exactly one prediction per request sample")
        result = predictions[0]
        if (result.get("episode_id") != episode_id or result.get("request_id") != request_id
                or result.get("noise_seed") != noise_seed):
            raise RuntimeError("Inference response identity or noise seed mismatch")
        try:
            actions = np.asarray(result["actions"], dtype=np.float64)
            normalized = np.asarray(result["normalized_actions"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidPolicyAction("Response has no numeric joint7 action chunk") from error
        if (actions.shape != (7, 7) or normalized.shape != (7, 7)
                or not np.isfinite(actions).all() or not np.isfinite(normalized).all()):
            raise InvalidPolicyAction("Expected finite seven-step seven-channel model actions")
        if not np.allclose(actions, normalized * SCALE, atol=2e-5, rtol=2e-6):
            raise RuntimeError("Server actions do not invert the declared joint7 normalization")
        evidence = {key: deepcopy(result[key]) for key in
                    ('normalized_state', 'initial_noise_sha256', 'preprocessed_image_sha256') if key in result}
        if 'normalized_state' in evidence and not np.allclose(joint7(evidence['normalized_state']),
                                                              sample['state'] / SCALE, atol=2e-7, rtol=2e-6):
            raise RuntimeError("Server normalized a different state than this observation")
        return actions, dict(model=deepcopy(model), raw_actions_deg_cm=actions.tolist(),
            normalized_actions=normalized.tolist(), round_trip_seconds=elapsed,
            inference_seconds=body.get("inference_seconds"), episode_id=episode_id,
            request_id=request_id, noise_seed=int(noise_seed), instruction=instruction,
            state_deg_cm=sample["state"], **evidence)


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
