#!/usr/bin/env python3
"""Exercise batched frozen inference over the real node transport, without Isaac."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pressb.online_rl.inference_transport import _pack
from pressb.rl_nodes.client import NodeClient
from benchmark_walloss_batch import load_observations, make_observations, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.count <= 64:
        parser.error("count must be in [1, 64]")
    sources, identities = load_observations([args.observation])
    observations = make_observations(sources, args.count, True)
    client = NodeClient(args.endpoint, timeout=600)
    report = dict(state="running", endpoint=args.endpoint, observations=args.count,
                  observation_files=identities, calls=[])
    contexts = []

    def call(method, payload=None):
        envelope = {"__pressb_rpc__": {"version": 1, "method": method.lstrip("/"), "payload": payload or {}}}
        started = time.perf_counter()
        result = client.call(method, payload)
        report["calls"].append(dict(method=method, seconds=time.perf_counter() - started,
                                    request_msgpack_bytes=len(_pack(envelope)),
                                    response_msgpack_bytes=len(_pack(result))))
        write_json(args.output, report)
        return result

    try:
        report["health_before"] = call("/health")
        metadata = report["health_before"]
        shape = (args.count, *metadata["noise"]["shape"])
        noise = np.random.default_rng(20261104).standard_normal(shape, dtype=np.float32)
        seeds = list(range(20261104, 20261104 + args.count))
        encoded = call("/encode", dict(observations=observations, seeds=seeds))["items"]
        contexts = [item["context_id"] for item in encoded]
        assert len(contexts) == args.count and len(set(contexts)) == args.count
        assert all(np.isfinite(item["features"]).all() for item in encoded)
        payload = {"items": [dict(context_id=key, initial_noise=value.tolist())
                             for key, value in zip(contexts, noise)]}
        decoded = call("/decode", payload)["items"]
        assert [item["context_id"] for item in decoded] == contexts
        actions = np.array([item["actions"] for item in decoded])
        expected_shape = (args.count, metadata["action"]["horizon"], len(metadata["action"]["names"]))
        assert actions.shape == expected_shape and np.isfinite(actions).all()
        repeated = np.array([item["actions"] for item in call("/decode", payload)["items"]])
        np.testing.assert_array_equal(actions, repeated)
        inferred = np.array([item["actions"] for item in call("/infer", dict(
            observations=observations, seeds=seeds, initial_noises=noise.tolist()))["items"]])
        np.testing.assert_array_equal(actions, inferred)
        released = call("/release", dict(context_ids=contexts))
        assert released["released"] == args.count
        contexts = []
        report.update(state="passed", action_shape=list(actions.shape),
                      repeated_decode_max_abs_error=float(np.max(np.abs(actions - repeated))),
                      infer_vs_cached_decode_max_abs_error=float(np.max(np.abs(actions - inferred))))
        report["health_after"] = call("/health")
        write_json(args.output, report)
        print(json.dumps({key: report[key] for key in ("state", "observations", "action_shape")}), flush=True)
    except Exception as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}")
        write_json(args.output, report)
        raise
    finally:
        if contexts:
            try:
                client.call("/release", dict(context_ids=contexts))
            except Exception:
                pass  # Failed transports are never retried; node TTL bounds retained contexts.
        client.close()


if __name__ == "__main__":
    main()
