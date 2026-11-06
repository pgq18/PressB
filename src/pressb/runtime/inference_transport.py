"""MessagePack wire primitives and the version-1 inference transport.

The NumPy encoding is interoperable with OpenPI. Application-specific legacy
policy adapters live in :mod:`pressb.compat.elevator_rl.inference_transport`.
"""
from __future__ import annotations

import base64
import threading

import numpy as np

from . import PROTOCOL_VERSION
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
