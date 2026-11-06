"""Legacy elevator inference HTTP handlers and cached model contexts."""
from __future__ import annotations

from collections import OrderedDict
import base64
from io import BytesIO
import math
import re
import time
import uuid

import numpy as np
from PIL import Image
import torch


class InferenceApplication:
    """Single-threaded HTTP handlers and a bounded, expiring GPU-context cache."""

    def __init__(self, backend, *, cache_size=256, cache_ttl=600.0, batch_size=1, clock=time.monotonic):
        if type(cache_size) is not int or cache_size < 1:
            raise ValueError("cache_size must be a positive integer")
        if not math.isfinite(cache_ttl) or cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive and finite")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.backend = backend
        self.cache_size = cache_size
        self.cache_ttl = float(cache_ttl)
        self.batch_size = batch_size
        self.clock = clock
        self.contexts = OrderedDict()
        self.encoded_count = self.decoded_count = 0

    def _purge(self):
        now = self.clock()
        for key in [key for key, (expires, _) in self.contexts.items() if expires <= now]:
            del self.contexts[key]

    @staticmethod
    def _fields(payload, allowed, required=None):
        if not isinstance(payload, dict) or set(payload) - (set(allowed) | {"protocol_version"}):
            raise ValueError("Unexpected request fields; expected a JSON object")
        if "protocol_version" in payload and (type(payload["protocol_version"]) is not int or payload["protocol_version"] != 1):
            raise ValueError("Incompatible protocol_version")
        if not set(allowed if required is None else required).issubset(payload):
            raise ValueError("Missing request fields")

    def health(self, payload=None):
        self._purge()
        backend_health = dict(self.backend.health())
        backend_health.update(encoder_batch_mode="batched" if self.batch_size > 1 else "serial",
                              microbatch_size=self.batch_size,
                              torch_threads=torch.get_num_threads(),
                              feature_encodings=["json", "float32_base64"],
                              accepted_image_sizes=[[640, 480], [224, 224]],
                              encode_release_context_ids=True)
        return dict(service="inference", ready=True, **backend_health,
                    cache=dict(size=len(self.contexts), capacity=self.cache_size, ttl_seconds=self.cache_ttl,
                               eviction="oldest_created", ttl_refresh_on_decode=False),
                    encoded_count=self.encoded_count, decoded_count=self.decoded_count)

    def _observation(self, observation):
        self._fields(observation, ("task", "state", "images", "control_state"), ("task", "state", "images"))
        if not isinstance(observation["task"], str) or not re.fullmatch(r"Press (?:2[4-9]|3[0-5]) floor\.", observation["task"]):
            raise ValueError("Expected an exact trained task for floors 24..35")
        state = observation["state"]
        if (not isinstance(state, list) or len(state) != 8
                or any(type(value) not in (int, float) for value in state)):
            raise ValueError("observation.state must be a numeric raw8 vector")
        state = np.asarray(state, dtype=np.float32)
        if not np.isfinite(state).all():
            raise ValueError("observation.state must be finite")
        self.backend.state_preparer(state)
        images = observation["images"]
        if not isinstance(images, dict) or set(images) != {"global", "wrist"}:
            raise ValueError("images must contain exactly global and wrist")
        decoded = {}
        for key, value in images.items():
            # WebSocket uses native RGB arrays or PNG bytes. The legacy HTTP
            # Base64 path retains the same validation and preprocessing.
            if isinstance(value, np.ndarray):
                if value.dtype != np.uint8 or value.shape not in ((480, 640, 3), (224, 224, 3)):
                    raise ValueError("Expected uint8 RGB image of size 640x480 or 224x224")
                decoded[key] = Image.fromarray(value.copy())
                continue
            if not isinstance(value, (str, bytes)) or len(value) > 12_000_000:
                raise ValueError("Expected a bounded base64 PNG image, PNG bytes, or RGB array")
            png = base64.b64decode(value, validate=True) if isinstance(value, str) else value
            with Image.open(BytesIO(png)) as picture:
                if picture.format != "PNG" or picture.size not in ((640, 480), (224, 224)) or picture.mode != "RGB":
                    raise ValueError("Expected a lossless RGB PNG of size 640x480 or 224x224")
                decoded[key] = picture.copy()
        for key, value in decoded.items():
            if isinstance(value, np.ndarray):
                value = Image.fromarray(value)
            if not isinstance(value, Image.Image):
                raise ValueError("decode_image must return RGB image data")
            decoded[key] = value
        return dict(task=observation["task"], state=state,
                    global_image=decoded["global"], wrist_image=decoded["wrist"])

    def _actions(self, actions):
        if actions.shape != (7, 9) or not np.isfinite(actions).all():
            raise RuntimeError("Frozen policy must return finite [7,9] actions")
        pose8 = self.backend.pose_converter(actions)
        return actions.tolist(), np.asarray(pose8).tolist()

    def encode(self, payload):
        self._fields(payload, ("observations", "seeds", "include_base_actions", "release_context_ids", "feature_encoding"),
                     ("observations", "seeds", "include_base_actions"))
        observations, seeds = payload["observations"], payload["seeds"]
        include = payload["include_base_actions"]
        if (not isinstance(observations, list) or not observations or len(observations) > self.cache_size
                or not isinstance(seeds, list) or len(seeds) != len(observations)
                or type(include) is not bool):
            raise ValueError("Expected 1..cache_size observations, one seed per sample and a boolean base-action flag")
        if any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
            raise ValueError("seeds must contain integers in [0, 2**63)")
        release_keys = payload.get("release_context_ids", [])
        if not isinstance(release_keys, list) or any(not isinstance(key, str) for key in release_keys):
            raise ValueError("release_context_ids must be a list of strings")
        feature_encoding = payload.get("feature_encoding", "json")
        if feature_encoding not in ("json", "float32_base64"):
            raise ValueError("feature_encoding must be json or float32_base64")
        decoded = [self._observation(observation) for observation in observations]
        pending = []
        # Finish the whole request before committing contexts, including pose conversion.
        encoded = []
        for offset in range(0, len(decoded), self.batch_size):
            if self.batch_size == 1:
                encoded.append(self.backend.encode(decoded[offset], seed=seeds[offset], include_base_actions=include))
            else:
                batch = self.backend.encode_batch(decoded[offset:offset + self.batch_size],
                    seeds=seeds[offset:offset + self.batch_size], include_base_actions=include)
                if len(batch) != len(decoded[offset:offset + self.batch_size]):
                    raise RuntimeError("Encoder returned an incorrect batch size")
                encoded.extend(batch)
        for context, base in encoded:
            key = uuid.uuid4().hex
            item = dict(context_id=key, state_pose9=context.state_pose9.tolist())
            if feature_encoding == "float32_base64":
                feature = np.ascontiguousarray(context.feature, dtype="<f4")
                if feature.shape != (self.backend.feature_dim,) or not np.isfinite(feature).all():
                    raise RuntimeError("Invalid frozen encoder feature")
                item.update(feature_f32_b64=base64.b64encode(feature.tobytes()).decode("ascii"),
                            feature_dim=len(feature))
            else:
                item["feature"] = context.feature.tolist()
            if include:
                item["base_actions_pose9"], item["base_actions_pose8"] = self._actions(base)
            pending.append((key, context, item))
        self._purge()
        for key in dict.fromkeys(release_keys):
            self.contexts.pop(key, None)
        while len(self.contexts) + len(pending) > self.cache_size:
            self.contexts.popitem(last=False)
        for key, context, _ in pending:
            self.contexts[key] = (self.clock() + self.cache_ttl, context)
        self.encoded_count += len(pending)
        return dict(items=[item for _, _, item in pending])

    def decode(self, payload):
        self._fields(payload, ("items",))
        items = payload["items"]
        if not isinstance(items, list) or not items or len(items) > self.cache_size:
            raise ValueError("items must contain 1..cache_size contexts")
        self._purge()
        requests = []
        for item in items:
            self._fields(item, ("context_id", "initial_noise"))
            key = item["context_id"]
            if not isinstance(key, str) or key not in self.contexts:
                raise ValueError("Context is missing or expired; encode the current observation explicitly")
            raw_noise = item["initial_noise"]
            if (not isinstance(raw_noise, list) or len(raw_noise) != 7
                    or any(not isinstance(row, list) or len(row) != 9
                           or any(type(value) not in (int, float) for value in row) for row in raw_noise)):
                raise ValueError("initial_noise must be a numeric [7,9] array")
            noise = np.asarray(raw_noise, dtype=np.float32)
            if not np.isfinite(noise).all():
                raise ValueError("initial_noise must be finite")
            requests.append((key, self.contexts[key][1], noise))
        decoded = []
        for offset in range(0, len(requests), self.batch_size):
            batch = requests[offset:offset + self.batch_size]
            if self.batch_size == 1:
                decoded.append(self.backend.decode(batch[0][1], batch[0][2]))
            else:
                actions = self.backend.decode_batch([item[1] for item in batch], [item[2] for item in batch])
                if len(actions) != len(batch):
                    raise RuntimeError("Decoder returned an incorrect batch size")
                decoded.extend(actions)
        result = []
        for (key, _, _), actions in zip(requests, decoded):
            actions9, actions8 = self._actions(actions)
            result.append(dict(context_id=key, actions_pose9=actions9, actions_pose8=actions8))
        self.decoded_count += len(result)
        return dict(items=result)

    def release(self, payload):
        self._fields(payload, ("context_ids",))
        keys = payload["context_ids"]
        if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys):
            raise ValueError("context_ids must be a list of strings")
        self._purge()
        released = sum(self.contexts.pop(key, None) is not None for key in dict.fromkeys(keys))
        return dict(released=released)

    @property
    def handlers(self):
        return {"/health": self.health, "/encode": self.encode, "/decode": self.decode, "/release": self.release}
