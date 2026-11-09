"""Native CPU thread settings without opening an Isaac application."""
from copy import deepcopy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.runtime.isaac_parallel import IsaacParallelWorld


def engine(task_close=lambda: None):
    return IsaacParallelWorld(SimpleNamespace(close=task_close),
                              SimpleNamespace(execution={}, num_envs=2))


def test_blas_limit_applies_to_loaded_numpy_pools_and_restores_them():
    from threadpoolctl import threadpool_info, threadpool_limits

    # Exercise a library already loaded before the engine's environment setup.
    np.linalg.solve(np.eye(3), np.ones(3))
    if not any(pool["user_api"] == "blas" for pool in threadpool_info()):
        pytest.skip("This NumPy build exposes no controllable BLAS pool")
    with threadpool_limits(limits=3, user_api="blas"):
        previous = {pool["filepath"]: pool["num_threads"] for pool in threadpool_info()
                    if pool["user_api"] == "blas"}
        runtime = engine()
        try:
            runtime._configure_blas()
            assert all(pool["num_threads"] == 1 for pool in runtime.blas_threadpools
                       if pool["user_api"] == "blas")
            assert all(pool["num_threads"] == 1 for pool in threadpool_info()
                       if pool["user_api"] == "blas")
        finally:
            runtime.close()
        restored = {pool["filepath"]: pool["num_threads"] for pool in threadpool_info()
                    if pool["user_api"] == "blas"}
        assert restored == previous
        assert runtime._blas_limit is None


def test_effective_pool_readback_records_blas_and_preserves_non_blas(monkeypatch):
    calls = []
    pools = [dict(user_api="blas", prefix="numpy_openblas", num_threads=32),
             dict(user_api="blas", prefix="scipy_openblas", num_threads=16),
             dict(user_api="openmp", prefix="libgomp", num_threads=8)]
    previous = deepcopy(pools)

    def restore():
        calls.append("restore")
        pools[:] = deepcopy(previous)

    def limits(*, limits, user_api):
        calls.append((limits, user_api))
        for pool in pools:
            if pool["user_api"] == user_api:
                pool["num_threads"] = limits
        return SimpleNamespace(restore_original_limits=restore)

    monkeypatch.setitem(sys.modules, "threadpoolctl", SimpleNamespace(
        threadpool_info=lambda: pools, threadpool_limits=limits))
    runtime = engine()
    runtime._configure_blas()
    assert calls == [(1, "blas")]
    assert [pool["num_threads"] for pool in runtime.blas_threadpools] == [1, 1, 8]
    pools[0]["prefix"] = "mutated"
    assert runtime.blas_threadpools[0]["prefix"] == "numpy_openblas"
    runtime.close()
    runtime.close()
    assert calls == [(1, "blas"), "restore"]
    assert pools == previous


def test_ineffective_blas_limit_is_detected_and_controller_remains_restorable(monkeypatch):
    restored = []
    controller = SimpleNamespace(restore_original_limits=lambda: restored.append(True))
    monkeypatch.setitem(sys.modules, "threadpoolctl", SimpleNamespace(
        threadpool_info=lambda: [dict(user_api="blas", num_threads=32)],
        threadpool_limits=lambda **kwargs: controller))
    runtime = engine()
    with pytest.raises(RuntimeError, match="did not apply"):
        runtime._configure_blas()
    assert runtime._blas_limit is controller
    runtime.close()
    assert restored == [True]


def test_cleanup_restores_blas_before_app_close_even_when_task_cleanup_fails():
    calls = []

    def fail():
        calls.append("task")
        raise RuntimeError("task cleanup failed")

    runtime = engine(fail)
    runtime._blas_limit = SimpleNamespace(restore_original_limits=lambda: calls.append("restore"))
    runtime.app = SimpleNamespace(close=lambda: calls.append("app"))
    with pytest.raises(RuntimeError, match="task cleanup failed"):
        runtime.close()
    assert calls == ["task", "restore", "app"]
    runtime.close()
    assert calls == ["task", "restore", "app"]


def test_app_still_closes_if_blas_restore_fails():
    calls = []

    def fail():
        calls.append("restore")
        raise RuntimeError("restore failed")

    runtime = engine()
    runtime._blas_limit = SimpleNamespace(restore_original_limits=fail)
    runtime.app = SimpleNamespace(close=lambda: calls.append("app"))
    with pytest.raises(RuntimeError, match="restore failed"):
        runtime.close()
    assert calls == ["restore", "app"]
    assert runtime._blas_limit is None


def test_execution_metadata_records_owned_effective_pool_settings():
    runtime = engine()
    runtime.config.update(render_resolution=[640, 480], renderer_profile="calibrated",
                          observation_render_frames=1, reset_render_frames=1, render_subframes=1)
    runtime.camera_names = ["up", "wrist"]
    runtime.image_preprocessing = None
    runtime.renderer_settings = {}
    runtime.directory = "/tmp/unused-settings-test"
    runtime.blas_threadpools = [dict(user_api="blas", num_threads=1)]
    metadata = runtime.execution_metadata()
    assert metadata["blas_thread_limit"] == 1
    assert metadata["blas_threadpools"] == runtime.blas_threadpools
    metadata["blas_threadpools"][0]["num_threads"] = 32
    assert runtime.blas_threadpools[0]["num_threads"] == 1
