"""Stream Torch checkpoints to an HTTP storage node without sender disk files.

The receiver writes only within its configured root. A complete SHA256-checked
upload is published atomically; interrupted/corrupt uploads leave no checkpoint.
Optional reads load into RAM for a learner whose home filesystem has no space.
No pickled checkpoint is deserialized by the storage server.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
from urllib.parse import urlsplit


MAX_CHECKPOINT_BYTES = 32 * 1024**3
CHUNK_BYTES = 1024**2
_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")


def _segment(value, name):
    if not isinstance(value, str) or not _SEGMENT.fullmatch(value) or ".." in value:
        raise ValueError(f"{name} must be a single safe path component")
    return value


def _destination(path):
    parts = path.split("/")
    if len(parts) != 4 or parts[:2] != ["", "checkpoints"]:
        raise ValueError("Expected /checkpoints/namespace/filename.pt")
    namespace, filename = (_segment(parts[2], "namespace"), _segment(parts[3], "filename"))
    if not filename.endswith(".pt"):
        raise ValueError("Checkpoint filename must end in .pt")
    return namespace, filename


def _url(value, *, base=False):
    url = urlsplit(value)
    if (url.scheme not in ("http", "https") or not url.hostname or url.username or url.password
            or url.query or url.fragment or base and url.path not in ("", "/")):
        raise ValueError("Checkpoint endpoint must be HTTP(S) without credentials/query/fragment")
    return url


@dataclass
class CheckpointSinkConfig:
    endpoint: str
    namespace: str
    timeout: float = 1800.
    max_bytes: int = MAX_CHECKPOINT_BYTES
    token_env: str = "PRESSB_CHECKPOINT_TOKEN"

    def __post_init__(self):
        _url(self.endpoint, base=True)
        _segment(self.namespace, "namespace")
        if type(self.timeout) not in (int, float) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Checkpoint timeout must be positive")
        if type(self.max_bytes) is not int or self.max_bytes <= 0:
            raise ValueError("Checkpoint max_bytes must be positive")
        if not isinstance(self.token_env, str) or not self.token_env:
            raise ValueError("Checkpoint token_env must name an environment variable")


def _connect(url, timeout):
    cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    return cls(url.hostname, url.port, timeout=timeout)


class _ChunkWriter:
    def __init__(self, connection, maximum):
        self.connection, self.maximum = connection, maximum
        self.size = 0
        self.digest = hashlib.sha256()

    def write(self, data):
        data = memoryview(data)
        if self.size + len(data) > self.maximum:
            raise ValueError("Checkpoint exceeds configured stream limit")
        for offset in range(0, len(data), CHUNK_BYTES):
            piece = data[offset:offset + CHUNK_BYTES]
            self.connection.send(f"{len(piece):x}\r\n".encode())
            self.connection.send(piece)
            self.connection.send(b"\r\n")
            self.digest.update(piece)
            self.size += len(piece)
        return len(data)

    def flush(self):
        pass

    def finish(self):
        self.connection.send((f"0\r\nX-Checkpoint-SHA256: {self.digest.hexdigest()}\r\n"
                              f"X-Checkpoint-Bytes: {self.size}\r\n\r\n").encode())


def save_checkpoint_stream(payload, filename, config: CheckpointSinkConfig | dict):
    """Serialize directly into an HTTP request and verify the durable receipt."""
    import torch
    cfg = config if isinstance(config, CheckpointSinkConfig) else CheckpointSinkConfig(**config)
    cfg.__post_init__()
    _segment(filename, "filename")
    route = f"/checkpoints/{cfg.namespace}/{filename}"
    _destination(route)
    url = _url(cfg.endpoint, base=True)
    connection = _connect(url, cfg.timeout)
    try:
        connection.putrequest("PUT", route)
        connection.putheader("Content-Type", "application/octet-stream")
        connection.putheader("Transfer-Encoding", "chunked")
        connection.putheader("Trailer", "X-Checkpoint-SHA256, X-Checkpoint-Bytes")
        token = os.environ.get(cfg.token_env)
        if token:
            connection.putheader("Authorization", "Bearer " + token)
        connection.endheaders()
        stream = _ChunkWriter(connection, cfg.max_bytes)
        torch.save(payload, stream)
        stream.finish()
        response = connection.getresponse()
        raw = response.read(65537)
        if response.status != 201 or len(raw) > 65536:
            raise RuntimeError(f"Checkpoint sink rejected upload (HTTP {response.status})")
        receipt = json.loads(raw)
        if (receipt.get("sha256") != stream.digest.hexdigest() or receipt.get("bytes") != stream.size
                or receipt.get("path") != route or receipt.get("committed") is not True):
            raise RuntimeError("Checkpoint receipt checksum/size/path mismatch")
        return {**receipt, "url": cfg.endpoint.rstrip("/") + route}
    finally:
        connection.close()


def load_checkpoint_stream(url, *, map_location="cpu", timeout=1800., max_bytes=MAX_CHECKPOINT_BYTES,
                           token_env="PRESSB_CHECKPOINT_TOKEN"):
    """Verified RAM-only checkpoint read; no remote/local cache file is created."""
    import torch
    parsed = _url(url)
    _destination(parsed.path)
    connection = _connect(parsed, timeout)
    try:
        token = os.environ.get(token_env)
        headers = {"Authorization": "Bearer " + token} if token else {}
        connection.request("GET", parsed.path, headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"Checkpoint read failed (HTTP {response.status})")
        size = int(response.getheader("Content-Length", "-1"))
        checksum = response.getheader("X-Checkpoint-SHA256", "")
        if not 0 < size <= max_bytes or not re.fullmatch(r"[a-f0-9]{64}", checksum):
            raise ValueError("Invalid checkpoint read length/checksum")
        digest, data = hashlib.sha256(), io.BytesIO()
        remaining = size
        while remaining:
            chunk = response.read(min(CHUNK_BYTES, remaining))
            if not chunk:
                raise IOError("Checkpoint response ended early")
            data.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        if digest.hexdigest() != checksum:
            raise ValueError("Checkpoint download checksum mismatch")
        data.seek(0)
        return torch.load(data, map_location=map_location, weights_only=True)
    finally:
        connection.close()


class CheckpointSinkServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, directory, *, token=None, max_bytes=MAX_CHECKPOINT_BYTES, timeout=1800.):
        if type(max_bytes) is not int or max_bytes <= 0 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Invalid checkpoint server size or timeout")
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.token, self.max_bytes, self.request_timeout = token, max_bytes, timeout
        self.upload_lock = threading.Lock()
        super().__init__(address, _Handler)

    def destination(self, route):
        namespace, filename = _destination(route)
        folder = self.directory / namespace
        folder.mkdir(exist_ok=True)
        if folder.is_symlink() or folder.resolve().parent != self.directory:
            raise ValueError("Checkpoint namespace cannot traverse the output directory")
        target = folder / filename
        if target.is_symlink():
            raise ValueError("Checkpoint target cannot be a symlink")
        return target


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(self.server.request_timeout)

    def log_message(self, *args):
        pass  # Never record URLs, request bodies or authorization headers.

    def _reply(self, status, body):
        data = json.dumps(body, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    def _authorized(self):
        if self.server.token and not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + self.server.token):
            self._reply(401, {"error": "Unauthorized"})
            return False
        return True

    def do_PUT(self):
        if not self._authorized():
            return
        temporary = None
        acquired = False

        def cleanup_upload():
            nonlocal temporary, acquired
            try:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
                    temporary = None
            finally:
                if acquired:
                    self.server.upload_lock.release()
                    acquired = False

        try:
            target = self.server.destination(self.path)
            if target.exists():
                self._reply(409, {"error": "Checkpoint already exists"})
                return
            if self.headers.get("Transfer-Encoding", "").lower() != "chunked" or self.headers.get("Content-Length") is not None:
                raise ValueError("Require chunked transfer encoding")
            acquired = self.server.upload_lock.acquire(blocking=False)
            if not acquired:
                self._reply(409, {"error": "Another checkpoint upload is active"})
                return
            digest, size = hashlib.sha256(), 0
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix="." + target.name + ".", suffix=".partial", delete=False) as stream:
                temporary = Path(stream.name)
                while True:
                    line = self.rfile.readline(80)
                    if not re.fullmatch(rb"[0-9a-fA-F]{1,16}\r\n", line):
                        raise ValueError("Invalid chunk header")
                    count = int(line.strip(), 16)
                    if count == 0:
                        break
                    if count > 8 * CHUNK_BYTES or size + count > self.server.max_bytes:
                        raise ValueError("Checkpoint exceeds chunk/file limit")
                    remaining = count
                    while remaining:
                        chunk = self.rfile.read(min(CHUNK_BYTES, remaining))
                        if not chunk:
                            raise ValueError("Incomplete checkpoint chunk")
                        stream.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                        remaining -= len(chunk)
                    if self.rfile.read(2) != b"\r\n":
                        raise ValueError("Invalid chunk terminator")
                trailers = {}
                for _ in range(3):
                    line = self.rfile.readline(256)
                    if line == b"\r\n":
                        break
                    name, separator, value = line.decode("ascii").partition(":")
                    if not separator or not line.endswith(b"\r\n") or name in trailers:
                        raise ValueError("Invalid checkpoint trailer")
                    trailers[name] = value.strip()
                else:
                    raise ValueError("Too many checkpoint trailers")
                if (set(trailers) != {"X-Checkpoint-SHA256", "X-Checkpoint-Bytes"}
                        or trailers["X-Checkpoint-SHA256"] != digest.hexdigest()
                        or trailers["X-Checkpoint-Bytes"] != str(size) or size == 0):
                    raise ValueError("Checkpoint stream checksum/size mismatch")
                stream.flush()
                os.fsync(stream.fileno())
            # Hard-link publication cannot overwrite an existing checkpoint,
            # including a competing process that wrote after the initial check.
            os.link(temporary, target)
            temporary.unlink()
            temporary = None
            directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            receipt = {"committed": True, "path": self.path, "bytes": size, "sha256": digest.hexdigest()}
            target.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
            # Receiving the response must mean the upload is fully finished:
            # a subsequent request may immediately reuse this receiver.
            cleanup_upload()
            self._reply(201, receipt)
        except (ValueError, UnicodeError, OSError) as error:
            cleanup_upload()
            try:
                self._reply(400, {"error": type(error).__name__})
            except OSError:
                pass
        finally:
            cleanup_upload()

    def do_GET(self):
        if not self._authorized():
            return
        if self.path == "/health":
            self._reply(200, {"ready": True, "max_bytes": self.server.max_bytes, "upload_active": self.server.upload_lock.locked()})
            return
        try:
            target = self.server.destination(self.path)
            if not target.is_file():
                self._reply(404, {"error": "Checkpoint not found"})
                return
            with target.open("rb") as stream:
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
                    digest.update(chunk)
                size = stream.tell()
                stream.seek(0)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(size))
                self.send_header("X-Checkpoint-SHA256", digest.hexdigest())
                self.send_header("Connection", "close")
                self.end_headers()
                for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
                    self.wfile.write(chunk)
                self.close_connection = True
        except (ValueError, OSError):
            self._reply(400, {"error": "Invalid checkpoint path/read"})
