"""Serial node hosting and config-loaded adapters; no model/Isaac imports here."""
from __future__ import annotations

import asyncio
import hmac
import importlib
import json
import os
from pathlib import Path
import signal

from .rpc import RPCServer, MAX_BODY_BYTES
from .inference_transport import _pack, _unpack


def load_backend(spec, kind=None):
    """Instantiate an explicit factory or a registered task/model plugin.

    Existing ``factory='module:function'`` configurations remain valid. Short
    plugin names require the service's ``task`` or ``model`` registry kind.
    """
    if not isinstance(spec, dict) or ("factory" in spec) == ("plugin" in spec):
        raise ValueError("backend must specify exactly one of factory or plugin and config={...}")
    config = spec.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("backend.config must be an object")
    if "plugin" in spec:
        plugin = spec["plugin"]
        if not isinstance(plugin, str) or not plugin.strip():
            raise ValueError("backend.plugin must be a nonempty registered name")
        if kind not in ("task", "model"):
            raise ValueError("backend.plugin requires kind='task' or kind='model'")
        from pressb.registry import resolve
        factory = resolve(kind, plugin)
    else:
        factory = spec["factory"]
    if not isinstance(factory, str):
        raise ValueError("backend.factory must be module:function")
    module, sep, name = factory.partition(":")
    if not sep or not module or not name:
        raise ValueError("backend.factory must be module:function")
    return getattr(importlib.import_module(module), name)(config)


def _message_limit(value):
    if type(value) is not int or value < 1:
        raise ValueError("max_message_bytes must be a positive integer")
    return value


async def serve_websocket(service, host, port, token=None, *, max_message_bytes=MAX_BODY_BYTES):
    max_message_bytes = _message_limit(max_message_bytes)
    # websockets>=14 lives in the isolated inference environment. The learner
    # and simulator's websockets12 clients remain wire-compatible.
    from websockets.asyncio.server import serve
    metadata = service.handlers["/health"]({})

    async def process_request(connection, request):
        if token and not hmac.compare_digest(request.headers.get("Authorization", ""), "Api-Key " + token):
            return connection.respond(401, "Authentication required\n")

    async def connection_handler(connection):
        await connection.send(_pack(metadata))
        async for frame in connection:
            try:
                if not isinstance(frame, bytes):
                    raise ValueError("Expected binary MessagePack request")
                message = _unpack(frame)
                envelope = message.get("__pressb_rpc__", {})
                if envelope.get("version") != 1 or not isinstance(envelope.get("payload"), dict):
                    raise ValueError("Expected version 1 PressB RPC envelope")
                route = "/" + envelope.get("method", "")
                if route not in service.handlers:
                    raise ValueError("Unknown node operation")
                # No await inside model/physics dispatch: all CUDA / Isaac calls
                # remain serialized on the owner thread, across connections too.
                result = service.handlers[route](envelope["payload"])
                result = {**result, "protocol_version": 1}
            except Exception as error:
                result = {"error": f"{type(error).__name__}: {error}", "protocol_version": 1}
            await connection.send(_pack(result))

    async with serve(connection_handler, host, port, process_request=process_request,
                     compression=None, max_size=max_message_bytes, ping_interval=None):
        await asyncio.Future()


def run_service(kind, config_path):
    backend_kinds = {"inference": "model", "simulation": "task", "real_robot": "task"}
    if kind not in backend_kinds:
        raise ValueError("Unknown node kind")
    config = json.loads(Path(config_path).read_text())
    allowed = {"backend", "server", "service"}
    if set(config) - allowed:
        raise ValueError(f"Unknown node config sections: {sorted(set(config)-allowed)}")
    server = config.get("server", {})
    host, port = server.get("host", "127.0.0.1"), int(server.get("port", 8100 if kind == "inference" else 8101))
    transport = server.get("transport", "ws" if kind == "inference" else "http")
    max_message_bytes = _message_limit(server.get("max_message_bytes", MAX_BODY_BYTES))
    token = os.environ.get(server["token_env"]) if server.get("token_env") else None
    if server.get("token_env") and not token:
        raise ValueError("Configured token_env is unset or empty")
    if transport not in ("http", "ws"):
        raise ValueError("server.transport must be http or ws")
    backend_spec = config["backend"]
    backend = service = None
    previous_sigterm = None
    physical_signal_handler = False
    def stop_service(signum, frame):
        raise KeyboardInterrupt
    if kind == "real_robot":
        # Service managers normally stop nodes with SIGTERM. Turn that into
        # the same unwind path as Ctrl-C so physical devices are closed.
        previous_sigterm = signal.signal(signal.SIGTERM, stop_service)
        physical_signal_handler = True
    try:
        # Preserve the one-argument factory hook used by existing service hosts.
        backend = (load_backend(backend_spec, kind=backend_kinds[kind])
                   if isinstance(backend_spec, dict) and "plugin" in backend_spec
                   else load_backend(backend_spec))
        if kind == "inference":
            from .inference import InferenceService
            service = InferenceService(backend, **config.get("service", {}))
        else:
            from .simulation import SimulationService
            service = SimulationService(backend, **config.get("service", {}))
        metadata = service.handlers["/health"]({})
        if kind == "real_robot" and metadata.get("environment_type") != "real_robot":
            raise ValueError("real_robot requires a physical task backend with environment_type='real_robot'")
        if metadata.get("environment_type") == "real_robot" and not physical_signal_handler:
            previous_sigterm = signal.signal(signal.SIGTERM, stop_service)
            physical_signal_handler = True
        print(json.dumps({"event": "ready", "endpoint": f"{transport}://{host}:{port}",
                          "metadata": metadata}, allow_nan=False), flush=True)
        if transport == "ws":
            asyncio.run(serve_websocket(service, host, port, token,
                                        max_message_bytes=max_message_bytes))
        else:
            with RPCServer((host, port), service.handlers, token=token) as http:
                http.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            close = getattr(service if service is not None else backend, "close", None)
            if close:
                close()
        finally:
            if physical_signal_handler:
                signal.signal(signal.SIGTERM, previous_sigterm)
