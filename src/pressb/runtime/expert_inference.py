"""Optional trainable action-expert operations for the shared inference service."""
from __future__ import annotations

from copy import deepcopy
import time

from .contract import validate_actions
from .expert_sync import ExpertWeights, expert_spec, tree_to_wire


def _row(tree, index):
    """Keep a batch dimension so model adapters can collate conditioning."""
    import torch
    if isinstance(tree, torch.Tensor):
        return tree[index:index + 1].detach().clone()
    if isinstance(tree, dict):
        return {key: _row(value, index) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_row(value, index) for value in tree)
    if isinstance(tree, list):
        return [_row(value, index) for value in tree]
    return tree


class ExpertInference:
    def __init__(self, service, config):
        from .denoising import GaussianChainPolicy

        if not isinstance(config, dict):
            raise ValueError("service.action_expert must be an object or null")
        creator = getattr(service.backend, "create_action_expert", None)
        if not callable(creator):
            raise ValueError("Model backend does not support action expert optimization")
        self.service = service
        self.config = deepcopy(config)
        self.expert = creator(deepcopy(config))
        self.expert.eval()
        self.policy = GaussianChainPolicy(self.expert)
        self.weights = ExpertWeights(self.expert, on_commit=self._invalidate)
        self.spec = expert_spec(self.expert)
        service.metadata["capabilities"] = list(dict.fromkeys([
            *service.metadata["capabilities"], "sample_chain", "action_expert_update"]))
        service.metadata.update(frozen=False, encoder_frozen=True,
                                learning_scope="action_expert")

    def _invalidate(self):
        self.service.contexts.clear()
        invalidate = getattr(self.expert, "invalidate_cache", None)
        if callable(invalidate):
            invalidate()

    def metadata(self):
        return {**self.weights.metadata(), "spec": deepcopy(self.spec), "config": deepcopy(self.config)}

    def _weight_operation(self, operation, payload, fields):
        with self.service._lock:
            self.service._ensure_open()
            self.service._fields(payload, set(fields), set(fields))
            return getattr(self.weights, operation)(payload)

    def sample_chain(self, payload):
        import torch

        service = self.service
        with service._lock:
            started = time.perf_counter()
            service._ensure_open()
            service._fields(payload, {"observations", "seeds", "expected_policy_version", "deterministic"},
                            {"observations", "seeds", "expected_policy_version"})
            self.weights._version(payload["expected_policy_version"])
            deterministic = payload.get("deterministic", False)
            if type(deterministic) is not bool:
                raise ValueError("deterministic must be boolean")
            observations, seeds = service._requests({key: payload[key] for key in ("observations", "seeds")})
            items = []
            # Bound encoder and chain activation memory by the same microbatch.
            with torch.no_grad():
                for offset in range(0, len(observations), service.batch_size):
                    batch = observations[offset:offset + service.batch_size]
                    batch_seeds = seeds[offset:offset + service.batch_size]
                    encoded = service._encode(batch, batch_seeds)
                    conditions = self.expert.pack_contexts([row[0] for row in encoded])
                    sample = self.policy.sample(conditions, batch_seeds, deterministic=deterministic)
                    for index, (_, features) in enumerate(encoded):
                        actions = validate_actions(sample["actions"][index].detach().float().cpu().numpy(),
                                                   service.metadata["action"])
                        items.append(dict(actions=actions.tolist(), features=features.tolist(),
                            chain=tree_to_wire(sample["chain"][index]),
                            logprobs=tree_to_wire(sample["logprobs"][index]) if sample["logprobs"] is not None else None,
                            conditioning=tree_to_wire(_row(conditions, index)),
                            policy_version=self.weights.version))
            service.encoded_count += len(items)
            service.decoded_count += len(items)
            return {"items": items, **self.weights.metadata(), "timing": {"total_seconds": time.perf_counter() - started}}

    def close(self):
        self.weights.close()

    @property
    def handlers(self):
        operations = {
            "begin": ("expected_version", "new_version", "size", "sha256", "weights_sha256"),
            "append": ("upload_id", "offset", "data"),
            "commit": ("upload_id",),
            "abort": ("upload_id",),
        }
        result = {"/sample_chain": self.sample_chain}
        for operation, fields in operations.items():
            result["/weights_" + operation] = lambda payload, op=operation, fs=fields: self._weight_operation(op, payload, fs)
        return result
