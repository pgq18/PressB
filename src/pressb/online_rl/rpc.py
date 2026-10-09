"""Small serial HTTP RPC server; handlers run on the caller's main thread.

This is deliberate for Isaac/Kit and CUDA context ownership. No automatic
client retry is allowed: a failed network response may follow completed steps.
"""
from __future__ import annotations

import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .protocol import PROTOCOL_VERSION

MAX_BODY_BYTES = 64 * 1024 * 1024


class RPCError(RuntimeError):
    pass


class RPCServer(HTTPServer):
    allow_reuse_address = True

    def __init__(self, address, handlers, token=None):
        self.handlers = dict(handlers)
        self.token = token
        super().__init__(address, _Handler)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Node-level structured logs describe transitions without enormous
        # per-camera access logs. HTTP failures are delivered to the caller.
        pass

    def _respond(self, code, value):
        body = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _dispatch(self, method):
        try:
            token = self.server.token
            if token and not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self._respond(401, {"error": "Authentication required"})
                return
            if self.path not in self.server.handlers or (method == "GET" and self.path not in ("/health", "/status")):
                self._respond(404, {"error": "Unknown route"})
                return
            payload = {}
            if method == "POST":
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY_BYTES:
                    raise ValueError("Invalid or oversized request body")
                payload = json.loads(self.rfile.read(size), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
                if not isinstance(payload, dict):
                    raise ValueError("Request must be a JSON object")
                if payload.get("protocol_version", PROTOCOL_VERSION) != PROTOCOL_VERSION:
                    raise ValueError("Incompatible protocol_version")
            result = self.server.handlers[self.path](payload)
            if not isinstance(result, dict):
                raise RuntimeError("RPC handler returned a non-object")
            self._respond(200, {**result, "protocol_version": PROTOCOL_VERSION})
        except (ValueError, KeyError, TypeError) as error:
            self._respond(400, {"error": f"{type(error).__name__}: {error}", "protocol_version": PROTOCOL_VERSION})
        except Exception as error:
            self._respond(500, {"error": f"{type(error).__name__}: {error}", "protocol_version": PROTOCOL_VERSION})

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")


class RPCClient:
    def __init__(self, endpoint, timeout=180., token=None):
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("Endpoint must be an HTTP(S) URL")
        self.endpoint = endpoint.rstrip("/")
        self.timeout = float(timeout)
        self.token = token

    def call(self, path, payload=None):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        data = None if payload is None else json.dumps({**payload, "protocol_version": PROTOCOL_VERSION}, allow_nan=False).encode()
        request = Request(self.endpoint + path, data=data, headers=headers, method="GET" if data is None else "POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            detail = error.read(4096).decode(errors="replace")
            raise RPCError(f"{path}: HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise RPCError(f"{path}: transport failed; execution state is unknown, request was not retried: {error}") from error
        if not isinstance(result, dict) or result.get("protocol_version") != PROTOCOL_VERSION:
            raise RPCError(f"{path}: invalid response protocol")
        return result
