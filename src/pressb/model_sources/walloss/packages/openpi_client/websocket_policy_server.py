import asyncio
import hmac
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Legacy inference is unchanged; split-inference RPC and API-key auth are opt-in.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        data_processor,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        rpc_handler=None,
        api_key: str | None = None,
    ) -> None:
        self._policy = policy
        self._data_processor = data_processor
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._rpc_handler = rpc_handler
        self._api_key = api_key
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._process_request,
        ) as server:
            await server.serve_forever()

    def _process_request(self, connection: _server.ServerConnection, request: _server.Request):
        health_response = _health_check(connection, request)
        if health_response is not None:
            return health_response
        is_websocket_upgrade = any(
            token.strip().lower() == "websocket"
            for value in request.headers.get_all("Upgrade")
            for token in value.split(",")
        )
        if self._api_key is not None and is_websocket_upgrade:
            authorization = request.headers.get_all("Authorization")
            expected = f"Api-Key {self._api_key}".encode("utf-8")
            if len(authorization) != 1 or not hmac.compare_digest(
                authorization[0].encode("utf-8"), expected
            ):
                return connection.respond(http.HTTPStatus.UNAUTHORIZED, "Unauthorized\n")
        return None

    def _dispatch_rpc(self, request):
        if not isinstance(request, dict):
            raise TypeError("__pressb_rpc__ must contain a dictionary")
        if type(request.get("version")) is not int or request["version"] != 1:
            raise ValueError("Unsupported PressB RPC version; expected 1")
        method = request.get("method")
        if not isinstance(method, str) or method not in {"encode", "decode", "release", "health"}:
            raise ValueError("Unsupported PressB RPC method")
        if not isinstance(request.get("payload"), dict):
            raise TypeError("PressB RPC payload must be a dictionary")
        response = self._rpc_handler(request)
        if not isinstance(response, dict):
            raise TypeError("PressB RPC handler must return a dictionary")
        return response

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                raw = msgpack_numpy.unpackb(await websocket.recv())

                # Extensions are opt-in. Unconfigured servers retain the legacy
                # observation and RTC paths, including their preprocessing.
                if self._rpc_handler is not None and isinstance(raw, dict) and "__pressb_rpc__" in raw:
                    close_after_response = False
                    try:
                        response = self._dispatch_rpc(raw["__pressb_rpc__"])
                        encoded_response = packer.pack(response)
                    except Exception as exc:
                        encoded_response = packer.pack({
                            "protocol_version": 1,
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                        })
                        close_after_response = not isinstance(exc, (ValueError, KeyError, TypeError))
                        if close_after_response:
                            logger.exception("PressB RPC handler failed")
                    await websocket.send(encoded_response)
                    if close_after_response:
                        await websocket.close(
                            code=websockets.frames.CloseCode.INTERNAL_ERROR,
                            reason="PressB RPC failed. Error included in previous frame.",
                        )
                        break
                    continue

                # Detect RTC protocol (backward-compatible)
                if isinstance(raw, dict) and "is_rtc" in raw:
                    obs = raw["obs"]
                    prev_action = raw.get("prev_action")
                    is_rtc = raw["is_rtc"]
                else:
                    obs = raw
                    prev_action = None
                    is_rtc = False

                obs = self._data_processor.in_process(obs)

                infer_time = time.monotonic()
                action = self._policy.infer(obs, prev_action=prev_action, is_rtc=is_rtc)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
