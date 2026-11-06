"""Legacy pose9 inference adapters and compatibility transport exports.

Generic wire primitives and clients are implemented by ``pressb.runtime``.
The legacy VLA-JEPA application and Wall-X hosting bridge remain here.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from pressb.model_sources import activate_source

from pressb.runtime import PROTOCOL_VERSION
from pressb.runtime.inference_transport import (
    InferenceClient, NAMESPACE, OPERATIONS, _pack, _pack_array, _unpack,
    _unpack_array, _wire_payload,
)
from pressb.runtime.rpc import MAX_BODY_BYTES, RPCClient, RPCError


class IdentityDataProcessor:
    """VLA-JEPA prepares its own raw TCP state; do not apply Wall-X min/max."""

    def in_process(self, observation):
        return observation


def _numbers(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [_numbers(item) for item in value]
    return value


class InferencePolicyAdapter:
    """Expose the frozen application through legacy infer and split RL calls.

    The legacy envelope is compatible; the checkpoint's state/action semantics
    are advertised rather than changed to Wall-X's seven-joint convention.
    """

    def __init__(self, application):
        self.application = application

    @property
    def metadata(self):
        return dict(action_dim=9, state_dim=8, pred_horizon=7,
                    camera_key=["face_view", "left_wrist_view"],
                    state_format="xyz_quaternion_wxyz_gripper_width_m",
                    action_format="absolute_tcp_xyz_rotation6d_rows",
                    camera_aliases=dict(face_view="global", left_wrist_view="wrist"),
                    rtc_config=dict(enabled=False),
                    pressb_rl=dict(version=PROTOCOL_VERSION, operations=list(OPERATIONS)))

    @staticmethod
    def _observation(obs):
        obs = dict(obs)
        for name in ("state", "control_state"):
            if name in obs:
                obs[name] = _numbers(obs[name])
        return obs

    def rpc_handler(self, envelope):
        if (not isinstance(envelope, dict) or set(envelope) != {"version", "method", "payload"}
                or type(envelope["version"]) is not int or envelope["version"] != PROTOCOL_VERSION):
            raise ValueError("Expected PressB RPC version 1 with method and payload")
        method, payload = envelope["method"], envelope["payload"]
        if not isinstance(method, str) or method not in OPERATIONS or not isinstance(payload, dict):
            raise ValueError("Unknown inference operation or invalid payload")
        payload = dict(payload)
        if payload.get("protocol_version", PROTOCOL_VERSION) != PROTOCOL_VERSION:
            raise ValueError("Incompatible protocol_version")
        if method == "encode":
            payload["observations"] = [self._observation(obs) for obs in payload["observations"]]
            payload["seeds"] = _numbers(payload["seeds"])
        elif method == "decode":
            payload["items"] = [{**item, "initial_noise": _numbers(item["initial_noise"])}
                                for item in payload["items"]]
        result = self.application.handlers["/" + method](payload)
        if method == "health":
            # The runner chooses JSON features; these are packed as ndarrays below.
            result = {**result, "feature_encodings": ["json"],
                      "feature_wire_format": "numpy", "transport": "websocket"}
        if "items" in result:
            result = {**result, "items": [dict(item) for item in result["items"]]}
            for item in result["items"]:
                for key in ("feature", "state_pose9", "base_actions_pose9", "base_actions_pose8",
                            "actions_pose9", "actions_pose8"):
                    if key in item:
                        item[key] = np.asarray(item[key], dtype=np.float32 if key == "feature" else None)
        return {**result, "protocol_version": PROTOCOL_VERSION}

    def infer(self, obs, prev_action=None, is_rtc=False):
        if is_rtc or prev_action is not None:
            raise ValueError("This VLA-JEPA checkpoint does not implement RTC; use is_rtc=False without prev_action")
        if not isinstance(obs, dict):
            raise ValueError("Observation must be a mapping")
        if "images" in obs:
            observation = {key: obs[key] for key in ("task", "state", "images", "control_state") if key in obs}
        else:
            observation = dict(task=obs["prompt"], state=obs["state"],
                               images={"global": obs["face_view"], "wrist": obs["left_wrist_view"]})
        seed = _numbers(obs.get("seed", 0))
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        prepared = self.application._observation(self._observation(observation))
        # A one-shot legacy inference needs no cached context. In particular,
        # it must not evict contexts held by a concurrent RL encode/decode user.
        backend = self.application.backend
        if self.application.batch_size == 1:
            _, actions = backend.encode(prepared, seed=seed, include_base_actions=True)
        else:
            _, actions = backend.encode_batch([prepared], seeds=[seed], include_base_actions=True)[0]
        actions9, _ = self.application._actions(actions)
        return {"action": np.asarray(actions9)[None]}


def load_wallx_server(wallx_repo=None):
    """Load the bundled Wall-X protocol server, or an explicit source override."""
    root = activate_source("walloss", wallx_repo, profile="protocol")
    packages = (root / "packages").resolve()
    source = packages / "openpi_client/websocket_policy_server.py"
    if not source.is_file():
        raise FileNotFoundError(f"Wall-X server missing: {source}; set --wall-x-repo")
    module = importlib.import_module("openpi_client.websocket_policy_server")
    if Path(module.__file__).resolve() != source:
        raise RuntimeError("A different openpi_client server is already imported; use a fresh process")
    return module.WebsocketPolicyServer
