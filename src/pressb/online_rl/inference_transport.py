"""Wall-X/OpenPI wire protocol for frozen VLA inference.

The reference server sends metadata first, then exchanges MessagePack frames.
Its NumPy wire format is different from the third-party msgpack_numpy package.
Only inference changes transport; Isaac continues to use the HTTP RPC service.
"""
from __future__ import annotations

import base64
import importlib
from pathlib import Path
import sys
import threading

import numpy as np

from .protocol import PROTOCOL_VERSION
from .rpc import MAX_BODY_BYTES, RPCClient, RPCError

OPERATIONS = ("health", "encode", "decode", "release")
NAMESPACE = "__pressb_rpc__"


def _pack_array(value):
    # Match ref/wall-x/packages/openpi_client/msgpack_numpy.py exactly on wire.
    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {value.dtype}")
        if isinstance(value, np.ndarray):
            return {b"__ndarray__": True, b"data": value.tobytes(),
                    b"dtype": value.dtype.str, b"shape": value.shape}
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _unpack_array(value):
    if b"__ndarray__" in value or b"__npgeneric__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {dtype}")
        if b"__ndarray__" in value:
            return np.ndarray(buffer=value[b"data"], dtype=dtype, shape=value[b"shape"])
        return dtype.type(value[b"data"])
    return value


def _pack(value):
    import msgpack
    return msgpack.packb(value, default=_pack_array)


def _unpack(value):
    import msgpack
    return msgpack.unpackb(value, object_hook=_unpack_array)


def _wire_payload(method, payload):
    """Remove Base64 overhead without decoding/re-encoding camera PNGs."""
    result = dict(payload or {})
    if method == "encode" and isinstance(result.get("observations"), list):
        result["observations"] = [dict(obs) for obs in result["observations"]]
        for obs in result["observations"]:
            if isinstance(obs.get("images"), dict):
                obs["images"] = {key: base64.b64decode(value, validate=True)
                                 if isinstance(value, str) else value
                                 for key, value in obs["images"].items()}
    return result


class InferenceClient:
    """Persistent WebSocket inference, with explicit legacy HTTP URL support.

    Requests are serialized even when the rollout worker changes threads. A
    failed connection is never retried or reconnected: cached contexts and the
    execution state of the last request may no longer be known.
    """

    def __init__(self, endpoint, timeout=180., token=None):
        if not endpoint.startswith(("http://", "https://", "ws://", "wss://")):
            raise ValueError("Inference endpoint must be an HTTP(S) or WS(S) URL")
        self.endpoint, self.timeout, self.token = endpoint.rstrip("/"), float(timeout), token
        self._http = RPCClient(endpoint, timeout, token) if endpoint.startswith(("http://", "https://")) else None
        self._ws = None
        self._failed = False
        self._lock = threading.Lock()
        self.metadata = None

    def _connect(self):
        from websockets.sync.client import connect
        headers = {"Authorization": "Api-Key " + self.token} if self.token else None
        # Isaac Sim pins websockets 12 in the shared learner environment. The
        # isolated inference server uses >=14; both speak the same wire protocol.
        self._ws = connect(self.endpoint, additional_headers=headers, compression=None,
                           max_size=MAX_BODY_BYTES, open_timeout=self.timeout, close_timeout=2)
        frame = self._ws.recv(timeout=self.timeout)
        if not isinstance(frame, bytes):
            raise RPCError("Expected Wall-X binary metadata handshake")
        self.metadata = _unpack(frame)
        capability = self.metadata.get("pressb_rl", {}) if isinstance(self.metadata, dict) else {}
        if (capability.get("version") != PROTOCOL_VERSION
                or not set(OPERATIONS).issubset(capability.get("operations", []))):
            raise RPCError("Server does not advertise PressB encode/decode capabilities; "
                           "a plain Wall-X infer server cannot serve the RL context protocol")

    def call(self, path, payload=None):
        if self._http:
            return self._http.call(path, payload)
        method = path.removeprefix("/")
        if method not in OPERATIONS:
            raise ValueError(f"Unknown inference operation: {path}")
        request = _pack({NAMESPACE: dict(version=PROTOCOL_VERSION, method=method,
                                        payload=_wire_payload(method, payload))})
        if len(request) > MAX_BODY_BYTES:
            raise ValueError("Inference request exceeds 64 MiB")
        with self._lock:
            if self._failed:
                raise RPCError("Inference connection failed or closed; request was not retried")
            try:
                if self._ws is None:
                    self._connect()
                self._ws.send(request)
                frame = self._ws.recv(timeout=self.timeout)
                if isinstance(frame, str):
                    raise RPCError(f"Error in inference server: {frame}")
                result = _unpack(frame)
                if not isinstance(result, dict) or result.get("protocol_version") != PROTOCOL_VERSION:
                    raise RPCError("Invalid inference response protocol")
            except Exception as error:
                self._failed = True
                if self._ws is not None:
                    self._ws.close()
                    self._ws = None
                raise RPCError(f"{path}: inference transport failed; execution state is unknown, "
                               f"request was not retried: {error}") from error
            if "error" in result:
                raise RPCError(f"{path}: {result['error']}")
            return result

    def close(self):
        with self._lock:
            if self._ws is not None:
                self._ws.close()
                self._ws = None
            self._failed = True


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
    """Load the actual server used by ref/wall-x/inference/launch_server.py."""
    root = Path(wallx_repo) if wallx_repo is not None else Path(__file__).resolve().parents[3] / "ref/wall-x"
    packages = (root / "packages").resolve()
    source = packages / "openpi_client/websocket_policy_server.py"
    if not source.is_file():
        raise FileNotFoundError(f"Wall-X server missing: {source}; set --wall-x-repo")
    sys.path.insert(0, str(packages))
    module = importlib.import_module("openpi_client.websocket_policy_server")
    if Path(module.__file__).resolve() != source:
        raise RuntimeError("A different openpi_client server is already imported; use a fresh process")
    return module.WebsocketPolicyServer
