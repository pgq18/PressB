"""CPU checks for canonical task modules and implementation provenance."""
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pressb.tasks.pap import backend, vector


@pytest.mark.parametrize("module,symbol", [
    ("pap.backend", "PAPParallelBackend"),
    ("pap.vector", "PAPVectorBackend"),
    ("pap.worker", "PAPEnvironment"),
    ("pap.recording", "PAPStateRecorder"),
    ("elevator.backend", "ElevatorBackend"),
])
def test_task_symbols_belong_to_the_canonical_implementation(module, symbol):
    implementation = importlib.import_module("pressb.tasks." + module)
    assert getattr(implementation, symbol).__module__ == implementation.__name__
    assert Path(implementation.__file__) == (
        backend.ROOT / "src/pressb/tasks" / (module.replace(".", "/") + ".py"))


def test_worker_module_help_does_not_initialize_isaac():
    module = "pressb.tasks.pap.worker"
    env = dict(os.environ, PYTHONPATH=str(backend.ROOT / "src"))
    result = subprocess.run([sys.executable, "-m", module, "--help"],
                            capture_output=True, text=True, env=env, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout and "--auth" in result.stdout


def test_pap_uses_the_shared_calibrated_camera_builders():
    from pressb.sensors import camera
    from pressb.tasks.pap import worker
    assert worker._intrinsics is camera.set_intrinsics
    assert worker._camera is camera.make_camera


@pytest.fixture
def cpu_vector_config(tmp_path, monkeypatch):
    trajectory = tmp_path / "episode_000000.npz"
    np.savez(trajectory, state_raw=np.zeros((1, 7)))
    from pressb.runtime import isaac_parallel
    from pressb.robots.piper.parallel import PiperJointDriver
    class MetadataEngine:
        def __init__(self, task, backend):
            self.robot = object.__new__(PiperJointDriver)
        def initialize(self): pass
        def close(self): pass
    monkeypatch.setattr(isaac_parallel, "IsaacParallelWorld", MetadataEngine)
    return {"python": sys.executable, "trajectory": str(trajectory),
            "num_envs": 1, "log_dir": str(tmp_path / "worker_logs")}


def test_repository_paths_and_provenance_follow_real_implementation(cpu_vector_config):
    assert backend.ROOT == Path(__file__).resolve().parents[1]
    instance = vector.PAPVectorBackend(cpu_vector_config)
    try:
        sources = instance.describe()["source_identity"]
        assert not any(name.startswith(("pick_place/", "table_buttons/", "rl_nodes/", "online_rl/"))
                       for name in sources)
        for name in ("tasks/pap/backend.py", "tasks/pap/worker.py", "tasks/pap/vector.py",
                     "tasks/pap/task.py", "tasks/pap/scene.py", "tasks/pap/evaluation.py",
                     "tasks/pap/initialization.py", "runtime/scene.py", "runtime/workbench.py",
                     "runtime/physics.py", "runtime/image_transport.py",
                     "runtime/parallel.py", "runtime/parallel_contract.py", "runtime/isaac_parallel.py",
                     "robots/piper/parallel.py", "tasks/pap/parallel.py",
                     "robots/piper/wrist_camera.py", "robots/piper/kinematics.py",
                     "control/joint_actions.py", "sensors/camera.py", "sensors/tiled.py"):
            assert name in sources
        assert "robots/piper/isaac.py" not in sources
        for name, identity in sources.items():
            contents = (backend.ROOT / "src/pressb" / name).read_bytes()
            assert identity == {"sha256": hashlib.sha256(contents).hexdigest(),
                                "bytes": len(contents)}
    finally:
        instance.close()


@pytest.mark.parametrize("changed_source", [
    "tasks/pap/backend.py", "tasks/pap/worker.py", "tasks/pap/vector.py",
    "runtime/physics.py", "runtime/image_transport.py",
    "runtime/parallel.py", "runtime/parallel_contract.py", "runtime/isaac_parallel.py",
    "robots/piper/parallel.py", "tasks/pap/parallel.py",
    "robots/piper/wrist_camera.py", "robots/piper/kinematics.py", "control/joint_actions.py",
    "sensors/camera.py", "sensors/tiled.py",
    "runtime/scene.py", "runtime/workbench.py", "tasks/pap/scene.py",
    "tasks/pap/task.py", "tasks/pap/evaluation.py", "tasks/pap/initialization.py",
])
def test_source_revision_detects_implementation_and_shared_dependency_changes(
        cpu_vector_config, monkeypatch, changed_source):
    first = vector.PAPVectorBackend(cpu_vector_config)
    original = first.describe()
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
        assert updated.describe()["source_revision"] != original["source_revision"]
        assert updated.describe()["task_config_sha256"] == original["task_config_sha256"]
    finally:
        updated.close()
