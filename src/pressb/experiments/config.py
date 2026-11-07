"""Resolve task/model/runner configuration without importing their dependencies.

Both simulated and physical task backends use ``nodes.simulation`` and the
trainer's ``endpoints.simulation`` key to share one environment wire contract.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re

from pressb.registry import resolve

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SECTIONS = {"schema_version", "name", "description", "extends", "task", "model", "nodes", "run", "collection"}
_VARIABLE = re.compile(r"\$\{([^{}]+)\}")


def _merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = (_merge(result[key], value) if isinstance(value, dict)
                       and isinstance(result.get(key), dict) else deepcopy(value))
    return result


def _expand(value, directory):
    if isinstance(value, dict):
        return {key: _expand(item, directory) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, directory) for item in value]
    if not isinstance(value, str):
        return value
    def replace(match):
        key = match[1]
        if key == "project_root":
            return str(PROJECT_ROOT)
        if key == "experiment_dir":
            return str(directory)
        if key.startswith("env:") and os.environ.get(key[4:]):
            return os.environ[key[4:]]
        raise ValueError(f"Unresolved experiment variable: {match[0]}")
    return _VARIABLE.sub(replace, value)


def _read(path, stack=()):
    path = Path(path).expanduser().resolve(strict=True)
    if path in stack:
        raise ValueError("Experiment inheritance cycle: " + " -> ".join(map(str, (*stack, path))))
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or set(raw) - _SECTIONS:
        raise ValueError(f"Invalid experiment sections in {path}")
    parent = raw.pop("extends", None)
    if parent is not None and (not isinstance(parent, str) or not parent):
        raise ValueError("extends must name one parent JSON file")
    base, sources = _read(path.parent / parent, (*stack, path)) if parent else ({}, [])
    # Resolve each file's variables before merging, preserving parent-relative paths.
    data = _merge(base, _expand(raw, path.parent))
    return data, [*sources, {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]


def _plugin(spec, kind):
    if (not isinstance(spec, dict) or set(spec) - {"plugin", "factory", "config"}
            or ("plugin" in spec) == ("factory" in spec)):
        raise ValueError(f"{kind} requires exactly one of plugin/factory, plus config")
    if not isinstance(spec.get("config", {}), dict):
        raise ValueError(f"{kind}.config must be an object")
    factory = resolve(kind, spec["plugin"]) if "plugin" in spec else spec["factory"]
    if not isinstance(factory, str) or not re.fullmatch(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*:[a-zA-Z_]\w*", factory):
        raise ValueError(f"{kind} factory must be module:function")
    return deepcopy(spec), factory


def load_experiment(path):
    data, sources = _read(path)
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("Experiment schema_version must be 1")
    if not isinstance(data.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", data["name"]):
        raise ValueError("Experiment name must be a nonempty file-safe identifier")
    task, task_factory = _plugin(data.get("task"), "task")
    model, model_factory = _plugin(data.get("model"), "model")
    nodes = data.get("nodes", {})
    if not isinstance(nodes, dict) or set(nodes) - {"inference", "simulation"}:
        raise ValueError("nodes supports inference and simulation only")
    compiled = {}
    for kind, backend, port, transport in (("inference", model, 8100, "ws"),
                                            ("simulation", task, 8101, "http")):
        node = nodes.get(kind, {})
        if not isinstance(node, dict) or set(node) - {"server", "service"}:
            raise ValueError(f"nodes.{kind} supports server/service only")
        server = {"host": "127.0.0.1", "port": port, "transport": transport, **node.get("server", {})}
        if (type(server["port"]) is not int or not 1 <= server["port"] <= 65535
                or server["transport"] not in ("ws", "http") or not isinstance(server["host"], str)):
            raise ValueError(f"Invalid nodes.{kind}.server address")
        if not isinstance(node.get("service", {}), dict):
            raise ValueError(f"nodes.{kind}.service must be an object")
        compiled[kind] = {"backend": backend, "server": server, "service": deepcopy(node.get("service", {}))}
    run = data.get("run", {})
    if not isinstance(run, dict):
        raise ValueError("run must be a trainer configuration object")
    run = deepcopy(run)
    run.setdefault("output_dir", str(PROJECT_ROOT / "outputs" / data["name"] / "run"))
    endpoints = run.setdefault("endpoints", {})
    if not isinstance(endpoints, dict):
        raise ValueError("run.endpoints must be an object")
    if endpoints.get("token"):
        raise ValueError("Do not store credentials in experiments; set PRESSB_RL_TOKEN at launch")
    for kind in ("inference", "simulation"):
        server = compiled[kind]["server"]
        host = server["host"] if server["host"] not in ("0.0.0.0", "::") else "127.0.0.1"
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        endpoints.setdefault(kind, f"{server['transport']}://{host}:{server['port']}")
    collection = data.get("collection", {})
    if (not isinstance(collection, dict) or set(collection) - {"directory", "include_images"}
            or type(collection.get("include_images", True)) is not bool):
        raise ValueError("collection supports directory and boolean include_images")
    compiled["trainer"] = run
    return {"name": data["name"], "experiment": data, "nodes": compiled,
            "factories": {"task": task_factory, "model": model_factory}, "sources": sources,
            "collection": deepcopy(collection)}


def materialize(experiment, directory):
    """Emit concrete three-node configs; never silently overwrite a different run."""
    directory = Path(directory).expanduser().resolve()
    documents = {f"{key}.json": value for key, value in experiment["nodes"].items()}
    documents["manifest.json"] = {"schema_version": 1, "name": experiment["name"],
        "factories": experiment["factories"], "sources": experiment["sources"],
        "configs_sha256": {name: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
                           for name, value in documents.items()}}
    # Preflight every destination before writing any file.
    for name, value in documents.items():
        target = directory / name
        if target.exists() and json.loads(target.read_text()) != value:
            raise FileExistsError(f"Refusing to overwrite different resolved config: {target}")
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in documents.items():
        target = directory / name
        if not target.exists():
            with target.open("x") as stream:
                stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return {key: directory / f"{key}.json" for key in experiment["nodes"]}
