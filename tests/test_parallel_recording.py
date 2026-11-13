"""Local recording preserves the wire scheduler's action/state/image boundaries."""
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from pressb.runtime.contract import decode_image
from pressb.runtime.isaac_parallel import IsaacParallelWorld
from pressb.runtime.parallel import SharedWorldBackend
from test_parallel_backend import DynamicsEngine, LiftTask, actions, reset_all


class RawEngine(DynamicsEngine):
    def __init__(self, task, backend):
        super().__init__(task, backend)
        self.raw_captures = []
        self.advance_on_capture = False

    def capture_rgb(self, env_ids, render_frames, reset_history=True):
        self.raw_captures.append((list(env_ids), self.world.count, render_frames))
        if self.advance_on_capture:
            self.world.current_time += self.task.execution['physics_dt']
        return {i: {'up': np.full((2, 3, 3), self.world.count, dtype=np.uint8)} for i in env_ids}


def make():
    backend = SharedWorldBackend(LiftTask(), engine_factory=RawEngine)
    reset_all(backend)
    return backend


def test_raw_step_preserves_next_action_alignment_without_png_or_extra_ticks():
    backend = make()
    initial_state = backend.engine.robot.q.copy()
    initial = backend.capture_for_recording([0, 1])
    wire_captures = len(backend.engine.captures)
    reply = backend.step_for_recording(actions(horizon=1))
    assert backend.engine.world.count == 4
    np.testing.assert_array_equal(initial_state, np.zeros((2, 2)))
    assert initial[0]['up'][0, 0, 0] == 0
    assert reply['rgb'][0]['up'][0, 0, 0] == 4
    np.testing.assert_allclose(reply['items'][0]['observation']['state'], [.9375, .0375])
    np.testing.assert_allclose(reply['items'][0]['info']['applied_actions'], [[1., .04]])
    assert reply['items'][0]['observation']['images'] == {}
    assert len(backend.engine.captures) == wire_captures
    assert backend.engine.raw_captures[-1] == ([0, 1], 4, 2)


def test_terminal_rgb_is_saved_before_peer_ticks_can_move_dead_environment():
    backend = make()
    backend.task.thresholds[0] = .01
    reply = backend.step_for_recording(actions(horizon=1))
    rows = {r['env_id']: r for r in reply['items']}
    assert rows[0]['terminated'] and not rows[1]['terminated']
    assert rows[0]['info']['physics_steps'] == 1
    assert rows[1]['info']['physics_steps'] == 4
    assert reply['rgb'][0]['up'][0, 0, 0] == 1
    assert reply['rgb'][1]['up'][0, 0, 0] == 4
    np.testing.assert_allclose(rows[0]['observation']['state'], [.5, .02])
    np.testing.assert_allclose(backend.engine.robot.q[0], [.9375, .0375])
    assert reply['snapshots'][0]['object_position'][2] == pytest.approx(.01)
    assert reply['snapshots'][1]['object_position'][2] == pytest.approx(.04)
    assert backend.engine.objects[0, 2] == pytest.approx(.04)
    assert reply['snapshots'][0]['episode_physics_steps'] == 1
    np.testing.assert_allclose(reply['snapshots'][0]['q'], rows[0]['observation']['state'])
    assert 'terminal_observation_omitted' not in rows[0]
    assert backend.engine.raw_captures == [([0], 1, 2), ([1], 4, 2)]


def test_recording_rejects_action_chunks_before_any_mutation():
    backend = make()
    with pytest.raises(ValueError, match='exactly one action'):
        backend.step_for_recording(actions(horizon=2))
    assert backend.engine.world.count == 0
    assert not backend.engine.raw_captures
    assert not backend.engine.robot.filter_history.any()


def test_raw_capture_checks_physical_clock_and_faults_if_renderer_advances_it():
    backend = make()
    backend.engine.advance_on_capture = True
    with pytest.raises(RuntimeError, match='advanced physical time'):
        backend.capture_for_recording([0])
    assert backend._fault.startswith('recording capture:')


def test_isaac_raw_tiles_are_owned_and_wire_pixels_match_after_output_resize(monkeypatch):
    settings = ModuleType('carb.settings')
    settings.get_settings = lambda: SimpleNamespace(set_bool=lambda *args: None)
    carb = ModuleType('carb')
    carb.settings = settings
    monkeypatch.setitem(__import__('sys').modules, 'carb', carb)
    monkeypatch.setitem(__import__('sys').modules, 'carb.settings', settings)
    engine = IsaacParallelWorld.__new__(IsaacParallelWorld)
    engine.config = dict(renderer_profile='calibrated', image_output_resolution=[2, 2])
    engine.camera_names = ['up', 'wrist']
    engine.image_preprocessing = None
    engine.backend = SimpleNamespace(timings={'encode_seconds': 0.})
    tiles = [np.full((4, 6, 3), i*31, dtype=np.uint8) for i in range(4)]
    calls = []
    def capture(*, return_tiles=True, subframes=None):
        calls.append(return_tiles)
        return tiles if return_tiles else None
    engine._capture = capture
    with ThreadPoolExecutor(max_workers=1) as pool:
        engine._pool = pool
        raw = engine.capture_rgb([1, 0], 2)
        wire = engine.capture([1, 0], 2)
    assert calls == [False, True, False, True]
    for eid in [1, 0]:
        for name in engine.camera_names:
            image = raw[eid][name]
            assert image.flags.owndata and image.flags.c_contiguous
            assert image.shape == (2, 2, 3)
            np.testing.assert_array_equal(np.asarray(decode_image(wire[eid][name])), image)
    tiles[2][:] = 255
    assert raw[1]['up'][0, 0, 0] == 62
