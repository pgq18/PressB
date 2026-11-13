"""CPU-only behavior tests; no renderer, camera, USD, or CUDA initialization."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.runtime.isaac_parallel import IsaacParallelWorld
from pressb.runtime.parallel import execution_config
from pressb.sensors.color_response import ColorResponse


def engine(workers=0):
    backend=SimpleNamespace(execution=execution_config({'color_response_workers':workers}),
                            num_envs=2,timings={})
    runtime=IsaacParallelWorld(SimpleNamespace(close=lambda:None),backend)
    runtime.camera_names=['up','wrist']
    runtime.camera_color_responses=[
        ColorResponse({'linear_rgb_gain':[.85,.96,2.3]}),
        ColorResponse({'linear_rgb_gain':[1.6,1.05,1.2],'saturation':.82})]
    runtime._configure_color_pool()
    return runtime


@pytest.mark.parametrize('bad',[-1,True,False,1.5,'4',None])
def test_color_pool_config_rejects_invalid_workers(bad):
    with pytest.raises(ValueError,match='color_response_workers'):
        execution_config({'color_response_workers':bad})


def test_color_pool_default_is_serial_and_never_creates_executor():
    runtime=engine()
    try:
        assert runtime.config['color_response_workers']==0
        assert runtime._color_pool is None
        owner=threading.get_ident()
        calls=[]
        def apply(rgb):
            calls.append(threading.get_ident())
            return rgb
        runtime.camera_color_responses=[SimpleNamespace(apply=apply)]*2
        images=[np.zeros((3,4,3),dtype=np.uint8)]*4
        assert runtime._apply_color_responses(images)==images
        assert calls==[owner]*4
    finally:
        runtime.close()


def test_parallel_color_pixels_order_input_storage_and_pool_reuse():
    serial,parallel=engine(),engine(4)
    images=[np.random.default_rng(i).integers(0,256,(12,16,3),dtype=np.uint8) for i in range(6)]
    before=[rgb.copy() for rgb in images]
    pool=parallel._color_pool
    try:
        for _ in range(2):
            expected=serial._apply_color_responses(images)
            actual=parallel._apply_color_responses(images)
            assert all(np.array_equal(a,b) for a,b in zip(actual,expected))
        assert all(np.array_equal(a,b) for a,b in zip(images,before))
        assert parallel._color_pool is pool
        assert parallel.backend.timings['color_response_images']==12
        assert parallel.backend.timings['color_response_seconds']>0
    finally:
        serial.close();parallel.close()
    assert parallel._color_pool is None
    assert pool._shutdown
    parallel.close()


def test_parallel_color_error_waits_for_running_borrowed_buffer_reads():
    runtime=engine(2)
    running,finished=threading.Event(),threading.Event()
    caller=threading.get_ident()
    workers=[]
    def failing(rgb):
        workers.append(threading.get_ident())
        assert running.wait(1.)
        raise ValueError('color failure')
    def delayed(rgb):
        workers.append(threading.get_ident())
        running.set()
        time.sleep(.01)
        finished.set()
        return rgb.copy()
    runtime.camera_color_responses=[SimpleNamespace(apply=failing),SimpleNamespace(apply=delayed)]
    try:
        with pytest.raises(ValueError,match='color failure'):
            runtime._apply_color_responses([np.zeros((2,2,3),np.uint8)]*2)
        assert finished.is_set()
        assert all(worker!=caller for worker in workers)
        assert runtime.backend.timings['color_response_images']==1
        assert runtime.backend.timings['color_response_seconds']>0
    finally:
        runtime.close()


def test_color_shutdown_failure_still_closes_png_pool_and_app():
    runtime=engine()
    calls=[]
    def fail(**kwargs):
        calls.append(('color',kwargs))
        raise RuntimeError('pool shutdown failure')
    runtime._color_pool=SimpleNamespace(shutdown=fail)
    runtime._pool=SimpleNamespace(shutdown=lambda **kw:calls.append(('png',kw)))
    runtime.app=SimpleNamespace(close=lambda:calls.append(('app',{})))
    with pytest.raises(RuntimeError,match='pool shutdown failure'):
        runtime.close()
    assert calls==[('color',{'wait':True,'cancel_futures':True}),('png',{'wait':True}),('app',{})]
    assert runtime._color_pool is None
    runtime.close()


@pytest.mark.parametrize('task_name',['playground','table_buttons','cube_stack','tabletop_cleanup'])
def test_task_configuration_keeps_shared_color_worker_option(task_name):
    import importlib
    class_name={'playground':'PlaygroundTask','table_buttons':'TableButtonsTask',
                'cube_stack':'CubeStackTask','tabletop_cleanup':'TabletopCleanupTask'}[task_name]
    task=getattr(importlib.import_module(f'pressb.tasks.{task_name}.parallel'),class_name)
    assert task({'color_response_workers':4}).execution['color_response_workers']==4
    assert task({}).execution['color_response_workers']==0


def test_pap_load_config_validates_color_workers(monkeypatch):
    from pressb.tasks.pap import parallel
    raw={'objects':[{}],'workers':[{'gpu':0}],
         'scene':{'up_camera':{'resolution':[640,480]},'wrist_camera':{'resolution':[640,480]}},
         'color_response_workers':4}
    monkeypatch.setattr(parallel,'load_pap_config',lambda config:{**raw,**config})
    assert parallel.load_config({})['color_response_workers']==4
    with pytest.raises(ValueError,match='color_response_workers'):
        parallel.load_config({'color_response_workers':True})
