"""Model-independent inference handlers with bounded, observation-owned caches.

The transport runs these handlers serially. The lock also prevents accidental
concurrent model access when embedded in another transport. A cache entry owns
the encoded observation and its seeded baseline noise; decoding never changes
the observation associated with another entry.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import math
import threading
import time
import uuid

from .contract import decode_image, finite_array, validate_actions, validate_metadata, validate_observation
from .image_transport import ValidatedObservation, validate_preprocessing_marker


class InferenceService:
    """Expose a frozen backend through the schema-v2 node API.

    Backends implement ``describe()``, ``encode_batch(observations, seeds)`` and
    ``decode_batch(contexts, initial_noises)``. Encoding returns one
    ``(private_context, feature_vector)`` pair per observation. ``None`` noise
    asks the backend for that context's reproducible seeded baseline.

    Cache TTL starts when a successful encode request is committed. Reads do
    not extend it. If capacity is exhausted, the oldest-created entries are
    evicted. Clients must release entries after storing their transitions and
    re-encode the observation if an entry has expired or been evicted.
    """

    def __init__(self, backend, *, cache_size=256, cache_ttl=600.0,
                 batch_size=1, clock=time.monotonic, collect_timing=False, action_expert=None):
        if type(cache_size) is not int or cache_size < 1:
            raise ValueError("cache_size must be a positive integer")
        if isinstance(cache_ttl, bool) or not isinstance(cache_ttl, (int, float)) \
                or not math.isfinite(cache_ttl) or cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive and finite")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(collect_timing) is not bool:
            raise ValueError("collect_timing must be a boolean")
        self.backend = backend
        self.metadata = deepcopy(validate_metadata(backend.describe(), node_type="inference"))
        self.metadata["capabilities"] = list(dict.fromkeys([
            *self.metadata.get("capabilities", []), "encode_release_contexts"]))
        self.cache_size = cache_size
        self.cache_ttl = float(cache_ttl)
        self.batch_size = batch_size
        # Native model batching is a backend capability; the actual microbatch
        # limit belongs to this service configuration, not the checkpoint.
        if self.metadata.get("batching", {}).get("native_model_batching") is True:
            self.metadata["batching"]["microbatch_size"] = batch_size
        self.clock = clock
        self.contexts = OrderedDict()
        self.encoded_count = 0
        self.decoded_count = 0
        self._lock = threading.RLock()
        self._closed = False
        self.collect_timing = collect_timing
        self.last_timing = {}
        self._timing = {}
        self.action_expert = None
        if action_expert is not None:
            from .expert_inference import ExpertInference
            self.action_expert = ExpertInference(self, action_expert)

    def _timed_result(self, result, started):
        if self.collect_timing:
            self._timing["total_seconds"] = time.perf_counter() - started
            # Host wall times include normal backend synchronization points;
            # profiling adds no CUDA synchronization to the critical path.
            self.last_timing = dict(self._timing)
            result["timing"] = dict(self._timing)
        return result

    @staticmethod
    def _fields(payload, allowed, required=()):
        if not isinstance(payload, dict):
            raise ValueError("request must be an object")
        # The HTTP/MessagePack envelope remains v1; the node contract is v2.
        if "protocol_version" in payload and (type(payload["protocol_version"]) is not int
                                              or payload["protocol_version"] != 1):
            raise ValueError("unsupported transport protocol_version")
        unexpected = set(payload) - set(allowed) - {"protocol_version"}
        if unexpected or not set(required).issubset(payload):
            raise ValueError(f"unexpected or missing request fields: allowed={sorted(allowed)}")

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("inference service is closed")

    def _purge(self):
        now = self.clock()
        for key in [key for key, (deadline, _) in self.contexts.items() if deadline <= now]:
            del self.contexts[key]

    def _requests(self, payload, *, infer=False):
        started = time.perf_counter()
        allowed = {"observations", "seeds", "initial_noises"} if infer else {
            "observations", "seeds", "include_base_actions", "release_context_ids"}
        self._fields(payload, allowed, {"observations", "seeds"})
        observations, seeds = payload["observations"], payload["seeds"]
        if not isinstance(observations, list) or not 1 <= len(observations) <= self.cache_size:
            raise ValueError("observations must contain 1..cache_size items")
        if not isinstance(seeds, list) or len(seeds) != len(observations) or any(
                type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
            raise ValueError("seeds must contain one integer in [0, 2**63) per observation")
        if not infer and "release_context_ids" in payload:
            keys = payload["release_context_ids"]
            if not isinstance(keys, list) or len(keys) > self.cache_size or any(
                    not isinstance(key, str) or not key for key in keys):
                raise ValueError("release_context_ids must be a bounded list of nonempty strings")
        observations = [validate_observation(item, self.metadata["observation"]) for item in observations]
        # Validate compressed payloads before any microbatch reaches the GPU.
        # Retain request-local decoded pixels so backend preflight and encode
        # do not decompress the same PNG three times. Never trust wire sidecars.
        prepared = []
        for observation in observations:
            images = {}
            for camera in self.metadata["observation"]["cameras"]:
                value = observation["images"][camera]
                image = decode_image(value, require_rgb=self.metadata.get("image_mode") == "RGB")
                validate_preprocessing_marker(value, image, self.metadata.get("image_preprocessing"))
                images[camera] = image
            prepared.append(ValidatedObservation(observation, images))
        observations = prepared
        validated_at = time.perf_counter()
        preflight = getattr(self.backend, "validate_observations", None)
        if callable(preflight):
            preflight(observations)
        if not infer and type(payload.get("include_base_actions", False)) is not bool:
            raise ValueError("include_base_actions must be a boolean")
        if self.collect_timing:
            self._timing["validation_seconds"] = validated_at - started
            self._timing["backend_preprocess_seconds"] = time.perf_counter() - validated_at
        return observations, seeds

    def _noise(self, value):
        if value is None:
            return None
        return finite_array(value, tuple(self.metadata["noise"]["shape"]), name="initial_noise").copy()

    def _encode(self, observations, seeds):
        started = time.perf_counter()
        encoded = []
        for offset in range(0, len(observations), self.batch_size):
            batch = observations[offset:offset + self.batch_size]
            result = list(self.backend.encode_batch(batch, seeds[offset:offset + self.batch_size]))
            if len(result) != len(batch):
                raise RuntimeError("backend returned an incorrect encoder batch size")
            for item in result:
                if not isinstance(item, (tuple, list)) or len(item) != 2:
                    raise RuntimeError("backend encoding must contain (context, features)")
                context, features = item
                features = finite_array(features, (self.metadata["feature_dim"],), name="features").copy()
                encoded.append((context, features))
        if self.collect_timing:
            self._timing["backend_encode_seconds"] = time.perf_counter() - started
        return encoded

    def _decode(self, contexts, noises):
        started = time.perf_counter()
        decoded = []
        for offset in range(0, len(contexts), self.batch_size):
            batch = contexts[offset:offset + self.batch_size]
            result = list(self.backend.decode_batch(batch, noises[offset:offset + self.batch_size]))
            if len(result) != len(batch):
                raise RuntimeError("backend returned an incorrect decoder batch size")
            decoded.extend(validate_actions(actions, self.metadata["action"]).copy() for actions in result)
        if self.collect_timing:
            self._timing["backend_decode_seconds"] = time.perf_counter() - started
        return decoded

    def health(self, payload=None):
        with self._lock:
            self._ensure_open()
            if payload is not None:
                self._fields(payload, set())
            self._purge()
            metadata = deepcopy(self.metadata)
            if self.action_expert is not None:
                metadata["action_expert"] = self.action_expert.metadata()
            return dict(metadata, ready=True,
                        cache=dict(size=len(self.contexts), capacity=self.cache_size,
                                   ttl_seconds=self.cache_ttl, eviction="oldest_created",
                                   ttl_refresh_on_decode=False),
                        microbatch_size=self.batch_size, encoded_count=self.encoded_count,
                        decoded_count=self.decoded_count)

    def encode(self, payload):
        with self._lock:
            started, self._timing = time.perf_counter(), {}
            self._ensure_open()
            observations, seeds = self._requests(payload)
            release_ids = list(dict.fromkeys(payload.get("release_context_ids", [])))
            encoded = self._encode(observations, seeds)
            actions = self._decode([context for context, _ in encoded], [None] * len(encoded)) \
                if payload.get("include_base_actions", False) else None
            # Commit the entire request only after all features/actions validate.
            pending = []
            for index, (context, features) in enumerate(encoded):
                key = uuid.uuid4().hex
                item = dict(context_id=key, features=features.tolist())
                if actions is not None:
                    item["base_actions"] = actions[index].tolist()
                pending.append((key, context, item))
            self._purge()
            # Retire only after every replacement feature/action validates.
            # Do this before eviction so another client's older cache entries
            # are not displaced by contexts this request already relinquished.
            for key in release_ids:
                self.contexts.pop(key, None)
            while len(self.contexts) + len(pending) > self.cache_size:
                self.contexts.popitem(last=False)
            deadline = self.clock() + self.cache_ttl
            for key, context, _ in pending:
                self.contexts[key] = (deadline, context)
            self.encoded_count += len(encoded)
            if actions is not None:
                self.decoded_count += len(actions)
            result = dict(items=[item for _, _, item in pending])
            if "release_context_ids" in payload:
                # Acknowledge unknown/expired IDs too: retirement is idempotent.
                result["released_context_ids"] = release_ids
            return self._timed_result(result, started)

    def decode(self, payload):
        with self._lock:
            started, self._timing = time.perf_counter(), {}
            self._ensure_open()
            self._fields(payload, {"items"}, {"items"})
            items = payload["items"]
            if not isinstance(items, list) or not 1 <= len(items) <= self.cache_size:
                raise ValueError("items must contain 1..cache_size entries")
            self._purge()
            requests = []
            for item in items:
                self._fields(item, {"context_id", "initial_noise"}, {"context_id"})
                key = item["context_id"]
                if not isinstance(key, str) or key not in self.contexts:
                    raise ValueError("context is missing or expired; encode the current observation again")
                requests.append((key, self.contexts[key][1], self._noise(item.get("initial_noise"))))
            decoded = self._decode([item[1] for item in requests], [item[2] for item in requests])
            self.decoded_count += len(decoded)
            return self._timed_result(dict(items=[dict(context_id=item[0], actions=actions.tolist())
                               for item, actions in zip(requests, decoded)]), started)

    def release(self, payload):
        with self._lock:
            self._ensure_open()
            self._fields(payload, {"context_ids"}, {"context_ids"})
            keys = payload["context_ids"]
            if not isinstance(keys, list) or len(keys) > self.cache_size or any(
                    not isinstance(key, str) for key in keys):
                raise ValueError("context_ids must be a bounded list of strings")
            self._purge()
            count = sum(self.contexts.pop(key, None) is not None for key in dict.fromkeys(keys))
            return dict(released=count)

    def infer(self, payload):
        with self._lock:
            started, self._timing = time.perf_counter(), {}
            self._ensure_open()
            observations, seeds = self._requests(payload, infer=True)
            raw_noises = payload.get("initial_noises", [None] * len(observations))
            if not isinstance(raw_noises, list) or len(raw_noises) != len(observations):
                raise ValueError("initial_noises must contain one entry per observation")
            noises = [self._noise(value) for value in raw_noises]
            encoded = self._encode(observations, seeds)
            decoded = self._decode([context for context, _ in encoded], noises)
            self.encoded_count += len(encoded)
            self.decoded_count += len(decoded)
            # One-shot requests do not create or evict persistent cache entries.
            return self._timed_result(dict(items=[dict(actions=actions.tolist()) for actions in decoded]), started)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self.contexts.clear()
            self._closed = True
            if self.action_expert is not None:
                self.action_expert.close()
            if callable(getattr(self.backend, "close", None)):
                self.backend.close()

    @property
    def handlers(self):
        handlers = {"/health": self.health, "/encode": self.encode, "/decode": self.decode,
                    "/release": self.release, "/infer": self.infer}
        if self.action_expert is not None:
            handlers.update(self.action_expert.handlers)
        return handlers
