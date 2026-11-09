"""Versioned action-expert transfer, independent of model and task families.

Tensor envelopes work with both JSON HTTP and MessagePack WebSocket. Weight
uploads are staged in bounded chunks; only a fully validated commit changes the
live policy. No shared filesystem between learner and inference is required.
"""
from __future__ import annotations

import base64
import hashlib
import io
import inspect
import math
from pathlib import Path
import tempfile
import time
import uuid


def tensor_to_wire(value):
    import torch
    if not isinstance(value, torch.Tensor):
        raise TypeError("Expected a tensor")
    tensor = value.detach().cpu().contiguous()
    data = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {"__tensor__": 1, "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape), "data": base64.b64encode(data).decode("ascii")}


def tree_to_wire(value):
    import torch
    if isinstance(value, torch.Tensor):
        return tensor_to_wire(value)
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Tensor tree dictionary keys must be strings")
        return {key: tree_to_wire(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"__tuple__": [tree_to_wire(item) for item in value]}
    if isinstance(value, list):
        return [tree_to_wire(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"Unsupported tensor tree value: {type(value).__name__}")


def tree_from_wire(value, device="cpu"):
    import torch
    if isinstance(value, dict) and "__tuple__" in value:
        if set(value) != {"__tuple__"} or not isinstance(value["__tuple__"], list):
            raise ValueError("Invalid tuple envelope")
        return tuple(tree_from_wire(item, device) for item in value["__tuple__"])
    if isinstance(value, dict) and "__tensor__" in value:
        if set(value) != {"__tensor__", "dtype", "shape", "data"} or value["__tensor__"] != 1:
            raise ValueError("Invalid tensor envelope")
        allowed = {"float16", "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool"}
        if value["dtype"] not in allowed:
            raise ValueError("Unsupported tensor dtype")
        shape = value["shape"]
        if not isinstance(shape, list) or len(shape) > 8 or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("Invalid tensor shape")
        raw = base64.b64decode(value["data"], validate=True)
        dtype = getattr(torch, value["dtype"])
        expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if len(raw) != expected:
            raise ValueError("Tensor shape does not match payload length")
        if expected == 0:
            return torch.empty(shape, dtype=dtype, device=device)
        result = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).clone()
        if result.is_floating_point() and not torch.isfinite(result).all():
            raise ValueError("Tensor contains nonfinite values")
        return result.to(device)
    if isinstance(value, dict):
        return {key: tree_from_wire(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_from_wire(item, device) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("Invalid tensor tree")


def state_sha256(state):
    """Canonical parameter identity, independent of torch ZIP metadata."""
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Action expert state must map names to tensors")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(str(tuple(tensor.shape)).encode() + b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def expert_spec(expert):
    """Bind sampler behavior to its implementation, independent of disk paths."""
    from . import denoising
    spec = dict(expert.describe())
    files = {"expert": inspect.getfile(type(expert)), "chain": denoising.__file__}
    spec["implementation_sha256"] = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                      for name, path in files.items()}
    return spec


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a SHA256 digest")
    return value


class ExpertWeights:
    """Staging owner; caller serializes all operations with inference's lock."""

    def __init__(self, expert, *, max_bytes=4 * 1024**3, ttl=600., on_commit=None):
        if type(max_bytes) is not int or max_bytes < 1 or not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("Invalid action expert upload limits")
        self.expert, self.max_bytes, self.ttl = expert, max_bytes, ttl
        self.on_commit = on_commit
        self.version = 0
        self.sha256 = state_sha256(expert.state_dict())
        self.pending = None

    def metadata(self):
        return {"policy_version": self.version, "weights_sha256": self.sha256}

    def _version(self, value):
        if type(value) is not int or value != self.version:
            raise ValueError(f"Stale policy version: expected {self.version}, received {value}")

    def close(self):
        if self.pending is not None:
            self.pending["file"].close()
            self.pending = None

    def begin(self, payload):
        self._version(payload["expected_version"])
        version, size = payload["new_version"], payload["size"]
        if type(version) is not int or version != self.version + 1:
            raise ValueError("new_version must increment policy_version by exactly one")
        if type(size) is not int or not 0 < size <= self.max_bytes:
            raise ValueError("Weight upload exceeds configured size limit")
        checksum = _digest(payload["sha256"], "sha256")
        weights_hash = _digest(payload["weights_sha256"], "weights_sha256")
        if self.pending is not None and time.monotonic() - self.pending["started"] < self.ttl:
            raise ValueError("Another weight upload is in progress")
        self.close()
        self.pending = dict(upload_id=uuid.uuid4().hex, version=version, size=size,
                            sha256=checksum, weights_sha256=weights_hash, offset=0,
                            started=time.monotonic(), file=tempfile.SpooledTemporaryFile(max_size=16 * 1024**2))
        return {"upload_id": self.pending["upload_id"], "offset": 0, **self.metadata()}

    def _pending(self, payload):
        pending = self.pending
        if pending is None or payload.get("upload_id") != pending["upload_id"]:
            raise ValueError("Unknown weight upload")
        if time.monotonic() - pending["started"] >= self.ttl:
            self.close()
            raise ValueError("Weight upload expired")
        return pending

    def append(self, payload):
        pending = self._pending(payload)
        if type(payload["offset"]) is not int or payload["offset"] != pending["offset"]:
            raise ValueError("Weight chunks must arrive once in sequential order")
        chunk = base64.b64decode(payload["data"], validate=True)
        if not 0 < len(chunk) <= 8 * 1024**2 or pending["offset"] + len(chunk) > pending["size"]:
            raise ValueError("Invalid weight chunk size")
        pending["file"].write(chunk)
        pending["offset"] += len(chunk)
        return {"upload_id": pending["upload_id"], "offset": pending["offset"]}

    def abort(self, payload):
        self._pending(payload)
        self.close()
        return {"aborted": True, **self.metadata()}

    def commit(self, payload):
        import torch
        pending = self._pending(payload)
        if pending["offset"] != pending["size"]:
            raise ValueError("Cannot commit an incomplete weight upload")
        stream = pending["file"]
        stream.seek(0)
        digest = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else None
        if digest is None:
            stream.seek(0)
            hasher = hashlib.sha256()
            for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
                hasher.update(chunk)
            digest = hasher.hexdigest()
        if digest != pending["sha256"]:
            raise ValueError("Weight upload checksum mismatch")
        stream.seek(0)
        candidate = torch.load(stream, map_location="cpu", weights_only=True)
        current = self.expert.state_dict()
        if not isinstance(candidate, dict) or set(candidate) != set(current):
            raise ValueError("Weight keys do not match this action expert")
        for name, target in current.items():
            tensor = candidate[name]
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != target.shape
                    or tensor.dtype != target.dtype or not torch.isfinite(tensor).all()):
                raise ValueError(f"Invalid weight tensor: {name}")
        # Sampler schedules and normalization are part of the declared model
        # contract, not PPO parameters. An upload cannot silently replace them.
        for name, buffer in self.expert.named_buffers(remove_duplicate=False):
            if name in candidate and not torch.equal(candidate[name], buffer.detach().cpu()):
                raise ValueError(f"Action expert buffers are immutable: {name}")
        checksum = state_sha256(candidate)
        if checksum != pending["weights_sha256"]:
            raise ValueError("Canonical action expert checksum mismatch")
        # Shape/dtype/finiteness failures never touch live weights. Retain a
        # rollback copy in case a device copy or backend invalidation fails.
        backup = {name: tensor.detach().cpu().clone() for name, tensor in current.items()}
        try:
            self.expert.load_state_dict(candidate, strict=True)
            if self.on_commit is not None:
                self.on_commit()
        except BaseException:
            self.expert.load_state_dict(backup, strict=True)
            raise
        self.version, self.sha256 = pending["version"], checksum
        self.close()
        return {"committed": True, **self.metadata()}


def publish_expert(client, state, expected_version, new_version=None, *, chunk_bytes=4 * 1024**2):
    """Send actual tensor bytes and wait for a matching atomic commit receipt."""
    import torch
    if type(chunk_bytes) is not int or not 0 < chunk_bytes <= 8 * 1024**2:
        raise ValueError("chunk_bytes must be in (0, 8 MiB]")
    state = {name: tensor.detach().cpu().clone() for name, tensor in state.items()}
    weights_hash = state_sha256(state)
    stream = io.BytesIO()
    torch.save(state, stream)
    blob = stream.getvalue()
    new_version = expected_version + 1 if new_version is None else new_version
    result = client.call("/weights_begin", dict(expected_version=expected_version, new_version=new_version,
                         size=len(blob), sha256=hashlib.sha256(blob).hexdigest(), weights_sha256=weights_hash))
    upload_id = result["upload_id"]
    for offset in range(0, len(blob), chunk_bytes):
        chunk = blob[offset:offset + chunk_bytes]
        receipt = client.call("/weights_append", dict(upload_id=upload_id, offset=offset,
                              data=base64.b64encode(chunk).decode("ascii")))
        if receipt.get("offset") != offset + len(chunk):
            raise RuntimeError("Inference did not acknowledge the complete weight chunk")
    receipt = client.call("/weights_commit", {"upload_id": upload_id})
    if (receipt.get("committed") is not True or receipt.get("policy_version") != new_version
            or receipt.get("weights_sha256") != weights_hash):
        raise RuntimeError("Inference did not acknowledge the requested action expert version")
    return receipt
