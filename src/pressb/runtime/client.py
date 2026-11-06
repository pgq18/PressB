"""HTTP and Wall-X-style MessagePack WebSocket node transport.

Failed requests are never retried automatically: physics may have advanced.
An inference connection's handshake does not substitute for schema negotiation.
"""
from __future__ import annotations

import math
import threading
import time

from .rpc import RPCClient, RPCError, MAX_BODY_BYTES
from .inference_transport import _pack, _unpack
from .image_transport import prepare_wire_payload


class NodeClient:
    def __init__(self, endpoint, timeout=180., token=None, *, max_message_bytes=MAX_BODY_BYTES,
                 image_transport="legacy", preprocess_images=False, image_workers=1):
        if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://", "ws://", "wss://")):
            raise ValueError("Node endpoint must be HTTP(S) or WS(S)")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if type(max_message_bytes) is not int or max_message_bytes < 1:
            raise ValueError("max_message_bytes must be a positive integer")
        if image_transport not in ("legacy", "auto"):
            raise ValueError("image_transport must be legacy or auto")
        if type(preprocess_images) is not bool:
            raise ValueError("preprocess_images must be a boolean")
        if type(image_workers) is not int or image_workers < 1:
            raise ValueError("image_workers must be a positive integer")
        self.endpoint, self.timeout, self.token = endpoint.rstrip("/"), timeout, token
        self.max_message_bytes = max_message_bytes
        self.image_transport, self.preprocess_images = image_transport, preprocess_images
        self.image_workers = image_workers
        self.last_call_timing = {}
        self._http = RPCClient(endpoint, timeout, token) if endpoint.startswith(("http://", "https://")) else None
        self._ws, self._failed, self._lock = None, False, threading.Lock()
        self.metadata = None

    def call(self, path, payload=None):
        with self._lock:
            if self._failed:
                raise RPCError("Node client is closed or failed; request was not retried")
            try:
                if self._http:
                    return self._http.call(path, payload)
                from websockets.sync.client import connect
                if self._ws is None:
                    headers = {"Authorization": "Api-Key " + self.token} if self.token else None
                    self._ws = connect(self.endpoint, additional_headers=headers, compression=None,
                                       max_size=self.max_message_bytes, open_timeout=self.timeout, close_timeout=2)
                    handshake = self._ws.recv(timeout=self.timeout)
                    if not isinstance(handshake, bytes):
                        raise RPCError("Expected a binary metadata handshake")
                    self.metadata = _unpack(handshake)
                    if not isinstance(self.metadata, dict) or self.metadata.get("schema_version") != 2:
                        raise RPCError("Server is not a configurable schema_version=2 node")
                started = time.perf_counter()
                prepared = prepare_wire_payload(path, payload or {}, self.metadata,
                                                image_transport=self.image_transport,
                                                preprocess_images=self.preprocess_images,
                                                image_workers=self.image_workers)
                prepared_at = time.perf_counter()
                request = _pack({"__pressb_rpc__": {"version": 1, "method": path.removeprefix("/"), "payload": prepared}})
                packed_at = time.perf_counter()
                if len(request) > self.max_message_bytes:
                    raise ValueError(f"Node request is {len(request)} bytes; exceeds "
                                     f"max_message_bytes={self.max_message_bytes}")
                self._ws.send(request)
                frame = self._ws.recv(timeout=self.timeout)
                if not isinstance(frame, bytes):
                    raise RPCError(f"Unexpected server response: {frame}")
                result = _unpack(frame)
                if not isinstance(result, dict) or result.get("protocol_version") != 1:
                    raise RPCError("Invalid node response protocol")
                if "error" in result:
                    raise RPCError(result["error"])
                self.last_call_timing = {"image_preprocess_seconds": prepared_at - started,
                                         "pack_seconds": packed_at - prepared_at,
                                         "exchange_seconds": time.perf_counter() - packed_at,
                                         "request_bytes": len(request), "response_bytes": len(frame)}
                return result
            except Exception:
                self._failed = True
                if self._ws is not None:
                    self._ws.close()
                    self._ws = None
                raise

    def close(self):
        with self._lock:
            if self._ws is not None:
                self._ws.close()
                self._ws = None
            self._failed = True
