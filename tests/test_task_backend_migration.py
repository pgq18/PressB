"""CPU checks for task-module compatibility and implementation provenance."""
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pressb.tasks.pap import backend, vector


@pytest.mark.parametrize("old,new", [
    ("pap", "pap.backend"),
    ("pap_vector", "pap.vector"),
    ("pap_worker", "pap.worker"),
    ("pap_recording", "pap.recording"),
    ("elevator", "elevator.backend"),
])
def test_legacy_task_import_is_the_implementation_module(old, new, monkeypatch):
    legacy = importlib.import_module("pressb.rl_nodes." + old)
    implementation = importlib.import_module("pressb.tasks." + new)
    assert legacy is implementation
    # Older tools patch module attributes; their patches must reach new callers.
    marker = object()
    monkeypatch.setattr(legacy, "migration_probe", marker, raising=False)
    assert implementation.migration_probe is marker


@pytest.mark.parametrize("module", ["pressb.rl_nodes.pap_worker", "pressb.tasks.pap.worker"])
def test_worker_module_help_does_not_initialize_isaac(module):
    env = dict(os.environ, PYTHONPATH=str(backend.ROOT / "src"))
    result = subprocess.run([sys.executable, "-m", module, "--help"],
                            capture_output=True, text=True, env=env, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout and "--auth" in result.stdout


def test_pap_uses_the_shared_calibrated_camera_builders():
    from pressb.runtime import isaac
    from pressb.tasks.pap import worker
    assert worker._intrinsics is isaac.set_intrinsics
    assert worker._camera is isaac.make_camera


@pytest.fixture
def cpu_vector_config(tmp_path, monkeypatch):
    trajectory = tmp_path / "episode_000000.npz"
    np.savez(trajectory, state_raw=np.zeros((1, 7)))
    monkeypatch.setattr(vector.PAPVectorBackend, "_initialize", lambda self: None)
    return {"python": sys.executable, "trajectory": str(trajectory),
            "num_envs": 1, "log_dir": str(tmp_path / "worker_logs")}


def test_repository_paths_and_provenance_follow_real_implementation(cpu_vector_config):
    assert backend.ROOT == Path(__file__).resolve().parents[1]
    instance = vector.PAPVectorBackend(cpu_vector_config)
    try:
        sources = instance.identity["source_identity"]
        for name in ("tasks/pap/backend.py", "tasks/pap/worker.py", "tasks/pap/vector.py",
                     "runtime/physics.py", "runtime/image_transport.py", "runtime/isaac.py",
                     "rl_nodes/pap.py", "rl_nodes/pap_worker.py", "rl_nodes/pap_vector.py"):
            contents = (backend.ROOT / "src/pressb" / name).read_bytes()
            assert sources[name] == {"sha256": hashlib.sha256(contents).hexdigest(),
                                     "bytes": len(contents)}
    finally:
        instance.close()


@pytest.mark.parametrize("changed_source", [
    "tasks/pap/backend.py", "tasks/pap/worker.py", "tasks/pap/vector.py",
    "runtime/physics.py", "runtime/image_transport.py", "runtime/isaac.py", "rl_nodes/pap_vector.py",
])
def test_source_revision_detects_implementation_and_shared_dependency_changes(
        cpu_vector_config, monkeypatch, changed_source):
    first = vector.PAPVectorBackend(cpu_vector_config)
    original = first.identity
    first.close()
    source_path = backend.ROOT / "src/pressb" / changed_source
    read_bytes = Path.read_bytes

    def changed_contents(path):
        contents = read_bytes(path)
        return contents + b"\n# simulated source update\n" if path == source_path else contents

    # Intercept reads without touching any source file or running simulation.
    monkeypatch.setattr(Path, "read_bytes", changed_contents)
    updated = vector.PAPVectorBackend(cpu_vector_config)
    try:
        assert updated.identity["source_revision"] != original["source_revision"]
        assert updated.identity["task_config_sha256"] == original["task_config_sha256"]
    finally:
        updated.close()
