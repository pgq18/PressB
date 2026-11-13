"""CPU behavior tests for the real shared scheduler and two different task hooks.

The injected engine integrates a simple servo and moving task objects. It owns
no scheduling, episode clocks, reward discounting, resets, or terminal logic;
those all run through the production SharedWorldBackend.
"""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.runtime.contract import decode_image, encode_image
from pressb.runtime.parallel import SharedWorldBackend
from pressb.runtime.parallel_contract import ParallelRobot, ParallelTask


class ServoRobot(ParallelRobot):
    def __init__(self, engine):
        self.engine = engine
        self.q = np.zeros((engine.num_envs, 2), dtype=np.float64)
        self.velocity = np.zeros_like(self.q)
        self.command = np.zeros_like(self.q)
        self.filter_history = np.zeros(engine.num_envs, dtype=np.int64)
        self.before_calls = []
        self.reset_calls = []
        self.cached = 0

    def cache_initial_state(self):
        self.cached += 1
        self.initial = self.q.copy()

    def reset(self, env_ids, specs):
        self.reset_calls.append(list(env_ids))
        for i in env_ids:
            self.q[i] = specs[i].get('initial_state_joint_rad_m', [0., 0.])
            self.command[i] = self.q[i]
            self.velocity[i] = 0
            self.filter_history[i] = 0

    def begin_action(self, env_ids, actions):
        for i, action in zip(env_ids, actions, strict=True):
            self.command[i] = action
            self.filter_history[i] += 1
        return {i: self.command[i].tolist() for i in env_ids}

    def before_tick(self, env_ids, substep, ticks_per_action):
        self.before_calls.append((tuple(env_ids), substep, ticks_per_action))

    def observations(self, env_ids):
        return {i: {'state': self.q[i].tolist(), 'control_state': self.command[i].tolist()}
                for i in env_ids}

    def diagnostic_state(self, env_ids):
        return {i: {'q': self.q[i].tolist(), 'velocity': self.velocity[i].tolist(),
                    'command': self.command[i].tolist(), 'filter': int(self.filter_history[i])}
                for i in env_ids}


class DynamicsEngine:
    def __init__(self, task, backend):
        self.task, self.backend = task, backend
        self.num_envs = task.execution['num_envs']
        self.world = SimpleNamespace(current_time=3., count=0)
        self.objects = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.local_ticks = np.zeros(self.num_envs, dtype=np.int64)
        self.captures, self.events = [], []
        self.capture_reset_flags = []
        self.history_resets = self.transform_publications = 0
        self.closed = False
        self.fail_tick = False
        self.fail_capture = False

    def initialize(self):
        self.robot = self.task.create_robot(self)
        self.task.bind(self)
        self.robot.cache_initial_state()
        self.task.after_settle()

    def tick(self, live):
        if self.fail_tick:
            raise RuntimeError('fixture physics failure')
        self.events.append(('physics', tuple(live)))
        self.world.count += 1
        self.world.current_time += self.task.execution['physics_dt']
        self.local_ticks[live] += 1
        # Dead objects still move in a shared world. Returned terminal state must
        # remain the state at the terminal tick, not this later measured state.
        self.objects[:, 2] += .01
        delta = (self.robot.command - self.robot.q) * .5
        self.robot.q += delta
        self.robot.velocity = delta / self.task.execution['physics_dt']

    def capture(self, env_ids, render_frames, reset_history=True):
        if self.fail_capture:
            raise RuntimeError('fixture renderer failure')
        self.captures.append(dict(ids=list(env_ids), render_frames=render_frames,
                                 world_tick=self.world.count, reset_history=reset_history))
        self.capture_reset_flags.append(getattr(self, '_reset_capture', False))
        return {i: {'up': encode_image(np.full((2, 3, 3), self.world.count % 256, dtype=np.uint8))}
                for i in env_ids}

    def reset_render_history(self):
        self.history_resets += 1

    def publish_transforms(self):
        self.transform_publications += 1

    def execution_metadata(self):
        return {'backend': 'injected_cpu_dynamics', 'shared_renderer': True}

    def check(self):
        if self.closed:
            raise RuntimeError('fixture engine closed')

    def close(self):
        self.closed = True


class LiftTask(ParallelTask):
    """A measured height condition, with physical-tick terminal boundaries."""
    def __init__(self, *, num_envs=2, **execution):
        self.config = {}
        self.execution = dict(num_envs=num_envs, gpu=0, physics_dt=.025,
            control_hz=10., horizon=4, max_episode_seconds=.3, settling_ticks=3,
            spacing_m=6., render_resolution=[3, 2], render_subframes=1,
            observation_render_frames=2, reset_render_frames=5,
            cpu_threads=1, png_workers=1, log_dir=None, gamma=.9,
            terminal_boundary='physics_tick', timeout_reason='timeout', headless=True)
        self.execution.update(execution)
        self.thresholds = np.full(num_envs, 99.)
        self.sample_calls = []
        self.reset_calls = []
        self.after_calls = []
        self.bad_reset_peer = False
        self.bad_reset_clock = False
        self.fail_reset = False

    def metadata(self):
        return dict(schema_version=2, node_type='simulation', task_id='cpu_lift',
            num_envs=self.execution['num_envs'],
            observation={'state': {'names': ['joint', 'gripper'], 'units': ['rad', 'm']}, 'cameras': ['up']},
            action={'representation': 'joint_position', 'frame': 'robot_base',
                'names': ['joint', 'gripper'], 'units': ['rad', 'm'],
                'horizon': self.execution['horizon'], 'control_hz': 10.,
                'lower': [-2., 0.], 'upper': [2., .1]},
            control_state_dim=2, task_features_dim=0,
            discount={'gamma': .9, 'timebase_hz': 10.}, reset_mode='indexed')

    def create_robot(self, engine):
        return ServoRobot(engine)

    def sample_reset(self, env_id, seed, options):
        self.sample_calls.append((env_id, seed, deepcopy(options)))
        allowed = {'offset', 'task', 'model_inputs', 'initial_state_joint_rad_m'}
        if set(options) - allowed:
            raise ValueError('Unknown fixture reset options')
        offset = options.get('offset', 0.)
        if not np.isfinite(offset):
            raise ValueError('Nonfinite fixture reset offset')
        return dict(seed=seed, x=float(np.random.default_rng(seed).uniform(-1, 1)+offset),
            task=options.get('task', 'Lift the object.'),
            model_inputs=deepcopy(options.get('model_inputs', {})),
            **({'initial_state_joint_rad_m': options['initial_state_joint_rad_m']}
               if 'initial_state_joint_rad_m' in options else {}))

    def reset_task(self, env_ids, specs):
        if self.fail_reset:
            raise RuntimeError('fixture reset failed')
        self.reset_calls.append(list(env_ids))
        for i in env_ids:
            self.engine.objects[i] = [specs[i]['x'], 0., 0.]
            self.engine.local_ticks[i] = 0
        if self.bad_reset_peer:
            peer = next(i for i in range(self.execution['num_envs']) if i not in env_ids)
            self.engine.objects[peer, 0] += 1
        if self.bad_reset_clock:
            self.engine.world.current_time += .025
        return {i: {'reset_x': specs[i]['x']} for i in env_ids}

    def before_tick(self, env_ids):
        self.engine.events.append(('task_before', tuple(env_ids)))

    def after_tick(self, env_ids):
        self.after_calls.append(tuple(env_ids))
        self.engine.events.append(('task_after', tuple(env_ids)))
        return {i: dict(success=bool(self.engine.objects[i, 2]+1e-12 >= self.thresholds[i]),
            terminated=bool(self.engine.objects[i, 2]+1e-12 >= self.thresholds[i]),
            reward=1. if self.engine.objects[i, 2]+1e-12 >= self.thresholds[i] else 0.,
            reason='success' if self.engine.objects[i, 2]+1e-12 >= self.thresholds[i] else None,
            info={'object_position': self.engine.objects[i].tolist()}) for i in env_ids}

    def diagnostic_state(self, env_ids):
        return {i: {'object_position': self.engine.objects[i].tolist(),
                    'local_ticks': int(self.engine.local_ticks[i])} for i in env_ids}


class PressTask(LiftTask):
    """A press latch can be vetoed by a later contact in the same interval."""
    def __init__(self, *, wrong_at=4, press_at=1, **kwargs):
        super().__init__(terminal_boundary='control_interval', **kwargs)
        self.wrong_at = wrong_at
        self.press_at = press_at
        self.pressed = np.zeros(self.execution['num_envs'], dtype=bool)
        self.wrong_contact = np.zeros_like(self.pressed)

    def reset_task(self, env_ids, specs):
        result = super().reset_task(env_ids, specs)
        self.pressed[env_ids] = self.wrong_contact[env_ids] = False
        return result

    def after_tick(self, env_ids):
        self.after_calls.append(tuple(env_ids))
        result = {}
        for i in env_ids:
            self.pressed[i] |= (self.engine.robot.q[i, 0] > .2
                               and self.engine.local_ticks[i] >= self.press_at)
            self.wrong_contact[i] |= self.engine.local_ticks[i] == self.wrong_at
            success = bool(self.pressed[i] and not self.wrong_contact[i])
            result[i] = dict(success=success, terminated=bool(self.pressed[i] or self.wrong_contact[i]),
                reward=float(success), reason='success' if success else 'non_target_contact',
                info={'target_pressed': bool(self.pressed[i]), 'wrong_contact': bool(self.wrong_contact[i])})
        return result


def make(task=None):
    task = task or LiftTask()
    return SharedWorldBackend(task, engine_factory=DynamicsEngine)


def reset_all(backend, seed=7):
    return backend.reset(list(range(backend.num_envs)), seed)


def actions(ids=(0, 1), horizon=2):
    return [{'env_id': i, 'actions': [[1., .04]]*horizon} for i in ids]


def rgb_tick(row):
    return int(np.asarray(decode_image(row['observation']['images']['up']))[0, 0, 0])


def test_batch_reset_matches_individual_seed_sampling_independent_of_order_and_peers():
    task = LiftTask(num_envs=3)
    backend = make(task)
    ids, seeds = [2, 0, 1], {'2': 123, 0: 456, '1': 789}
    options = {'offset': .2, 'by_env': {'0': {'offset': -.4, 'model_inputs': {'v': [1]}}}}
    original = deepcopy(options)
    rows = backend.reset(ids, 99999, options, env_seeds=seeds)
    expected = {i: task.sample_reset(i, int(seeds.get(str(i), seeds.get(i))),
        {**{'offset': .2}, **original['by_env'].get(str(i), {})}) for i in ids}
    assert {i: backend.reset_specs[i] for i in ids} == expected
    assert [r['env_id'] for r in rows] == ids
    assert task.reset_calls == [ids] and options == original
    another = make(LiftTask(num_envs=3))
    another.reset([0, 1, 2], 0, options, env_seeds=seeds)
    assert [another.reset_specs[i] for i in range(3)] == [backend.reset_specs[i] for i in range(3)]


@pytest.mark.parametrize('change', [dict(env_seeds={0: 1}), dict(env_seeds={0: 1, '0': 2, 1: 3}),
    dict(env_seeds={0: 1, 1: True}), dict(options={'by_env': []}),
    dict(options={'by_env': {'1': {'offset': float('nan')}}})])
def test_invalid_last_reset_spec_or_seed_has_no_physical_or_render_side_effect(change):
    backend = make()
    with pytest.raises((TypeError, ValueError)):
        backend.reset([0, 1], 3, **change)
    assert backend.engine.world.count == 0
    assert backend.task.reset_calls == backend.engine.robot.reset_calls == []
    assert backend.engine.captures == []


def test_indexed_reset_is_zero_tick_and_preserves_peer_state_clock_and_filter():
    backend = make()
    reset_all(backend)
    backend.step(actions(horizon=1))
    peer_before = deepcopy(backend.engine.robot.diagnostic_state([1]))
    task_before = deepcopy(backend.task.diagnostic_state([1]))
    ticks = backend.ticks.copy()
    clock = backend.engine.world.current_time
    world_ticks = backend.engine.world.count
    backend.end_episode([0])
    before_captures = len(backend.engine.captures)
    backend.reset([0], 123, {'initial_state_joint_rad_m': [.1, .02]})
    assert backend.engine.world.current_time == clock
    assert backend.engine.world.count == world_ticks
    assert backend.ticks.tolist() == [0, ticks[1]]
    assert backend.engine.robot.diagnostic_state([1]) == peer_before
    assert backend.task.diagnostic_state([1]) == task_before
    assert backend.engine.robot.filter_history.tolist() == [0, 1]
    assert backend.engine.captures[before_captures:] == [dict(ids=[0], render_frames=5,
        world_tick=world_ticks, reset_history=True)]


@pytest.mark.parametrize('bad', ['peer', 'clock'])
def test_task_reset_cannot_silently_modify_peer_or_advance_shared_clock(bad):
    backend = make()
    reset_all(backend)
    backend.end_episode([0])
    setattr(backend.task, 'bad_reset_' + bad, True)
    with pytest.raises(RuntimeError):
        backend.reset([0], 99)


def test_physical_terminal_snapshot_precedes_survivor_and_is_not_a_live_view():
    task = LiftTask(max_episode_seconds=.075)
    task.thresholds[0] = .01
    backend = make(task)
    reset_all(backend)
    rows = backend.step(actions(ids=(1, 0), horizon=2))
    late, early = rows
    assert [r['env_id'] for r in rows] == [1, 0]
    assert early['terminated'] and not early['truncated']
    assert late['truncated'] and not late['terminated']
    assert rgb_tick(early) == 1 and rgb_tick(late) == 3
    assert early['info']['object_position'][2] == pytest.approx(.01)
    assert backend.engine.objects[0, 2] == pytest.approx(.03)
    assert early['observation']['state'][0] == pytest.approx(.5)
    assert backend.ticks.tolist() == [1, 3]
    assert early['info']['physics_steps'] == 1 and late['info']['physics_steps'] == 3
    assert early['discount'] == pytest.approx(.9**.25)
    before = deepcopy(early)
    backend.reset([0, 1], 8)
    assert early == before


@pytest.mark.parametrize('event_tick', [1, 2, 4, 5, 8])
def test_physics_tick_reward_and_discount_use_actual_fractional_control_duration(event_tick):
    task = LiftTask(num_envs=1)
    task.thresholds[0] = event_tick*.01
    backend = make(task)
    reset_all(backend)
    row, = backend.step(actions(ids=(0,), horizon=3))
    assert row['info']['physics_steps'] == event_tick
    assert row['info']['elapsed_control_steps'] == event_tick/4
    assert row['reward'] == pytest.approx(.9**((event_tick-1)/4))
    assert row['discount'] == pytest.approx(.9**(event_tick/4))


@pytest.mark.parametrize('wrong_at,success', [(4, False), (99, True)])
def test_control_interval_checks_last_tick_even_after_first_tick_success(wrong_at, success):
    task = PressTask(num_envs=1, wrong_at=wrong_at)
    backend = make(task)
    reset_all(backend)
    row, = backend.step(actions(ids=(0,), horizon=3))
    assert len(task.after_calls) == 4
    assert backend.engine.world.count == row['info']['physics_steps'] == 4
    assert row['info']['success'] is success
    assert row['info']['target_pressed']
    assert row['info']['wrong_contact'] is (not success)
    assert row['reward'] == float(success)  # First full interval has gamma exponent zero.
    assert row['discount'] == pytest.approx(.9)
    assert rgb_tick(row) == 4
    assert [x[1] for x in backend.engine.robot.before_calls] == [0, 1, 2, 3]


def test_state_only_terminal_preserves_exact_measured_state_and_skips_terminal_render():
    task = LiftTask(num_envs=1)
    task.thresholds[0] = .02
    backend = make(task)
    reset_all(backend)
    backend.engine.captures.clear()
    row, = backend.step(actions(ids=(0,)), terminal_observation_mode='state_only_if_no_bootstrap')
    assert row['terminal_observation_omitted'] is True
    assert row['observation']['images'] == {}
    assert row['observation']['state'][0] == pytest.approx(.75)
    assert row['info']['physics_steps'] == 2
    assert row['reward'] == pytest.approx(.9**.25)
    assert backend.engine.captures == []


@pytest.mark.parametrize('mode', ['full', 'state_only_if_no_bootstrap'])
def test_live_slots_always_receive_current_rgb(mode):
    backend = make()
    reset_all(backend)
    backend.engine.captures.clear()
    rows = backend.step(actions(horizon=1), terminal_observation_mode=mode)
    assert all(not row['terminated'] and not row['truncated'] for row in rows)
    assert all(rgb_tick(row) == 4 for row in rows)
    assert backend.engine.captures[-1]['ids'] == [0, 1]


@pytest.mark.parametrize('case', ['missing_active', 'duplicate', 'unequal', 'nan', 'out_of_bounds', 'bad_mode'])
def test_complete_action_batch_validates_before_robot_targets_or_physics(case):
    backend = make()
    reset_all(backend)
    command, kwargs = actions(), {}
    if case == 'missing_active': command.pop()
    elif case == 'duplicate': command[1]['env_id'] = 0
    elif case == 'unequal': command[1]['actions'].pop()
    elif case == 'nan': command[1]['actions'][0] = [float('nan'), .02]
    elif case == 'out_of_bounds': command[1]['actions'][0] = [3., .02]
    else: kwargs['terminal_observation_mode'] = 'unknown'
    with pytest.raises(ValueError):
        backend.step(command, **kwargs)
    assert backend.engine.world.count == 0
    assert not backend.engine.robot.before_calls
    assert backend.engine.robot.filter_history.tolist() == [0, 0]


def test_fused_reset_retains_old_terminal_and_captures_survivor_with_new_episode_once():
    task = LiftTask()
    task.thresholds[0] = .01
    backend = make(task)
    reset_all(backend)
    backend.engine.captures.clear()
    reply = backend.step_and_reset(actions(horizon=1), reset_seed=17, reset_options={}, max_active_envs=2)
    terminal, live = reply['items']
    fresh, = reply['reset_items']
    assert terminal['terminal_observation_omitted'] and terminal['observation']['images'] == {}
    assert terminal['observation']['state'][0] == pytest.approx(.5)
    assert live['observation']['state'][0] == pytest.approx(.9375)
    assert fresh['env_id'] == 0 and fresh['observation']['state'] == [0., 0.]
    assert fresh['observation'] is not terminal['observation']
    assert backend.ticks.tolist() == [0, 4]
    assert backend.engine.world.count == 4
    assert len(backend.engine.captures) == 1
    assert backend.engine.captures[0]['ids'] == [0, 1]
    assert backend.engine.captures[0]['render_frames'] == 5
    assert rgb_tick(fresh) == rgb_tick(live) == 4


@pytest.mark.parametrize('with_reset', [False, True])
def test_fused_capture_selects_reset_render_settings_only_when_new_episode_is_present(with_reset):
    task = LiftTask(reset_render_subframes=4)
    if with_reset:
        task.thresholds[0] = .01
    backend = make(task)
    reset_all(backend)
    assert backend.engine.capture_reset_flags == [True]
    backend.engine.capture_reset_flags.clear()
    backend.step_and_reset(actions(horizon=1), reset_seed=17, reset_options={}, max_active_envs=2)
    assert backend.engine.capture_reset_flags == [with_reset]
    assert backend.engine._reset_capture is False


@pytest.mark.parametrize('budget,expected', [(0, []), (1, [1]), (2, [1, 0])])
def test_fused_all_terminal_honors_command_order_and_active_budget(budget, expected):
    task = LiftTask(max_episode_seconds=.075)
    task.thresholds[0] = .01
    backend = make(task)
    reset_all(backend)
    backend.engine.captures.clear()
    reply = backend.step_and_reset(actions(ids=(1, 0)), reset_seed=17, reset_options={}, max_active_envs=budget)
    assert [r['env_id'] for r in reply['reset_items']] == expected
    assert [r['info']['physics_steps'] for r in reply['items']] == [3, 1]
    assert backend.engine.world.count == 3
    assert len(backend.engine.captures) == bool(expected)


@pytest.mark.parametrize('changes', [dict(reset_seed=True), dict(reset_options={'by_env': []}),
    dict(reset_options={'by_env': {'1': {'offset': float('nan')}}}),
    dict(max_active_envs=True), dict(max_active_envs=3)])
def test_fused_prevalidates_all_possible_reset_specs_even_when_no_reset_expected(changes):
    backend = make()
    reset_all(backend)
    before = deepcopy(backend.engine.robot.diagnostic_state([0, 1]))
    kwargs = dict(reset_seed=17, reset_options={}, max_active_envs=0)
    kwargs.update(changes)
    with pytest.raises((ValueError, TypeError)):
        backend.step_and_reset(actions(horizon=1), **kwargs)
    assert backend.engine.world.count == 0
    assert backend.engine.robot.diagnostic_state([0, 1]) == before


def test_fused_reset_failure_returns_no_partial_success_and_stops_before_render():
    task = LiftTask(num_envs=1)
    task.thresholds[0] = .01
    backend = make(task)
    reset_all(backend)
    backend.engine.captures.clear()
    task.fail_reset = True
    with pytest.raises(RuntimeError, match='reset failed'):
        backend.step_and_reset(actions(ids=(0,)), reset_seed=17, reset_options={}, max_active_envs=1)
    assert backend.engine.world.count == 1
    assert backend.engine.captures == []


def test_end_episode_only_retires_selected_slot_without_tick_render_or_reset():
    backend = make()
    reset_all(backend)
    before = deepcopy(backend.engine.robot.diagnostic_state([0, 1]))
    captures = deepcopy(backend.engine.captures)
    backend.end_episode([0])
    assert backend.ready.tolist() == [False, True]
    assert backend.engine.world.count == 0 and backend.engine.captures == captures
    assert backend.engine.robot.diagnostic_state([0, 1]) == before
    row, = backend.step(actions(ids=(1,), horizon=1))
    assert row['env_id'] == 1 and backend.ticks.tolist() == [0, 4]


def test_profiling_status_is_detached_and_does_not_tick_or_render():
    backend = make()
    backend.reset([0, 1], 42)
    before = (backend.engine.world.count, len(backend.engine.captures))
    backend.timings['contact_callback_seconds'] = 1.25
    report = backend.status()
    assert report['ready_env_ids'] == [0, 1]
    assert report['timings']['contact_callback_seconds'] == 1.25
    report['timings']['contact_callback_seconds'] = -1
    report['ready_env_ids'].clear()
    assert backend.status()['timings']['contact_callback_seconds'] == 1.25
    assert backend.status()['ready_env_ids'] == [0, 1]
    assert (backend.engine.world.count, len(backend.engine.captures)) == before
    backend.close()
    assert backend.status()['closed'] is True


def test_close_is_idempotent_and_does_not_advance_physics():
    backend = make()
    reset_all(backend)
    backend.close()
    backend.close()
    assert backend.engine.closed and backend.engine.world.count == 0


@pytest.mark.parametrize('press_at', [5, 8])
def test_second_control_interval_success_is_checked_at_boundary_and_discounted_once(press_at):
    task = PressTask(num_envs=1, press_at=press_at, wrong_at=99)
    backend = make(task)
    reset_all(backend)
    row, = backend.step(actions(ids=(0,), horizon=3))
    assert row['info']['success']
    assert len(task.after_calls) == row['info']['physics_steps'] == 8
    assert len(row['info']['applied_actions']) == 2
    assert row['info']['elapsed_control_steps'] == 2
    assert row['reward'] == pytest.approx(.9)
    assert row['discount'] == pytest.approx(.9**2)
    assert rgb_tick(row) == 8


def test_control_interval_timeout_is_at_configured_boundary_and_not_partial_interval():
    task = PressTask(num_envs=1, press_at=99, wrong_at=99, max_episode_seconds=.2)
    backend = make(task)
    reset_all(backend)
    row, = backend.step(actions(ids=(0,), horizon=3))
    assert row['truncated'] and not row['terminated']
    assert not row['info']['success']
    assert row['info']['termination_reason'] == 'timeout'
    assert len(task.after_calls) == row['info']['physics_steps'] == 8
    assert row['reward'] == 0 and row['discount'] == pytest.approx(.9**2)
    assert rgb_tick(row) == 8


@pytest.mark.parametrize('changes', [
    {'control_hz': 11.}, {'physics_dt': 0.}, {'num_envs': True},
    {'gamma': 1.01}, {'gamma': float('nan')}, {'horizon': 0},
    {'terminal_boundary': 'unknown'},
    {'terminal_boundary': 'control_interval', 'max_episode_seconds': .075},
])
def test_invalid_execution_settings_fail_before_constructing_engine(changes):
    task = LiftTask()
    task.execution.update(changes)
    calls = []
    def factory(*args):
        calls.append(args)
        raise AssertionError('Invalid configuration constructed an engine')
    with pytest.raises(ValueError):
        SharedWorldBackend(task, engine_factory=factory)
    assert calls == []


def test_initial_reset_requires_all_slots_before_any_robot_or_task_reset():
    backend = make()
    with pytest.raises(ValueError, match='Initial reset'):
        backend.reset([0], 7)
    assert backend.engine.robot.reset_calls == backend.task.reset_calls == []
    assert backend.engine.captures == [] and backend.engine.world.count == 0
    reset_all(backend)
    assert backend.ready.tolist() == [True, True]


def test_robot_specific_action_validation_checks_entire_chunk_before_first_target():
    backend = make()
    reset_all(backend)
    def reject_last_target(command):
        if command[-1][0] > .9:
            raise ValueError('fixture robot domain invalid')
    backend.engine.robot.validate_actions = reject_last_target
    with pytest.raises(ValueError, match='robot domain'):
        backend.step(actions())
    assert backend.engine.robot.filter_history.tolist() == [0, 0]
    assert backend.engine.world.count == 0
    # Validation failures are recoverable and did not fault an untouched world.
    backend.engine.robot.validate_actions = lambda command: None
    assert len(backend.step(actions(horizon=1))) == 2


@pytest.mark.parametrize('failure', ['physics', 'renderer', 'task_outcome'])
def test_partial_step_failure_latches_fault_and_rejects_later_actions_and_reset(failure):
    backend = make()
    reset_all(backend)
    if failure == 'physics':
        backend.engine.fail_tick = True
    elif failure == 'renderer':
        backend.engine.fail_capture = True
    else:
        backend.task.after_tick = lambda ids: {i: {'reward': float('nan')} for i in ids}
    with pytest.raises((ValueError, RuntimeError)):
        backend.step(actions(horizon=1))
    world_count = backend.engine.world.count
    target_count = backend.engine.robot.filter_history.copy()
    reset_count = len(backend.engine.robot.reset_calls)
    for request in (lambda: backend.step(actions(horizon=1)), lambda: reset_all(backend)):
        with pytest.raises(RuntimeError, match='faulted'):
            request()
    assert backend.engine.world.count == world_count
    np.testing.assert_array_equal(backend.engine.robot.filter_history, target_count)
    assert len(backend.engine.robot.reset_calls) == reset_count


def test_wrong_thread_cannot_enter_physics_or_render():
    backend = make()
    reset_all(backend)
    captures = len(backend.engine.captures)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match='owning main thread'):
            executor.submit(backend.step, actions(horizon=1)).result()
    assert backend.engine.world.count == 0 and len(backend.engine.captures) == captures
    assert backend.engine.robot.filter_history.tolist() == [0, 0]


def test_metadata_describes_parallel_contract_without_mutable_aliases():
    backend = make(PressTask())
    metadata = backend.describe()
    assert metadata['num_envs'] == 2 and metadata['initial_reset'] == 'all_envs'
    assert metadata['scheduling'] == 'all_active'
    assert metadata['equal_execution_horizon'] is True
    assert metadata['execution_horizons'] == [1, 2, 3, 4]
    assert set(metadata['capabilities']) >= {'per_env_reset_seeds', 'terminal_state_only', 'step_and_reset'}
    assert metadata['execution']['indexed_reset_physics_ticks'] == 0
    assert metadata['execution']['terminal_boundary'] == 'control_interval'
    assert metadata['execution']['physics_hz'] == 40.
    metadata['execution']['indexed_reset_physics_ticks'] = 99
    assert backend.describe()['execution']['indexed_reset_physics_ticks'] == 0


class ReadOnlyRecorder:
    def __init__(self):
        self.resets, self.ticks, self.steps = [], [], []
        self.closed = 0
        self.events = []

    def on_reset(self, backend, ids, rows):
        self.events.append('reset')
        self.resets.append((backend.engine.world.count, list(ids), deepcopy(rows)))

    def after_tick(self, backend, ids, outcomes):
        self.events.append('tick')
        self.ticks.append(dict(world_tick=backend.engine.world.count, ids=list(ids),
            states=deepcopy(backend.engine.robot.observations(ids)),
            outcomes={i: deepcopy(outcomes[i]) for i in ids}))

    def after_step(self, backend, rows):
        self.events.append('step')
        self.steps.append((backend.engine.world.count, deepcopy(rows)))

    def close(self):
        self.events.append('close')
        self.closed += 1


def test_read_only_observer_receives_exact_terminal_tick_and_does_not_add_physics_or_render():
    def configured():
        task = LiftTask(max_episode_seconds=.075)
        task.thresholds[0] = .01
        return make(task)
    observed, control = configured(), configured()
    recorder = ReadOnlyRecorder()
    observed.add_observer(recorder)
    for backend in (observed, control):
        reset_all(backend)
    rows = observed.step(actions(ids=(1, 0), horizon=2))
    reference = control.step(actions(ids=(1, 0), horizon=2))
    assert rows == reference
    assert observed.engine.world.count == control.engine.world.count == 3
    assert observed.engine.world.current_time == control.engine.world.current_time
    assert observed.engine.captures == control.engine.captures
    assert len(recorder.resets) == len(recorder.steps) == 1
    assert len(recorder.ticks) == 3
    assert recorder.ticks[0]['ids'] == [1, 0]
    assert recorder.ticks[1]['ids'] == recorder.ticks[2]['ids'] == [1]
    assert recorder.ticks[0]['outcomes'][0]['terminated']
    assert recorder.ticks[0]['outcomes'][0]['info']['object_position'][2] == pytest.approx(.01)
    assert recorder.ticks[0]['states'][0]['state'][0] == pytest.approx(.5)
    assert recorder.ticks[2]['outcomes'][1]['truncated']
    assert recorder.ticks[2]['outcomes'][1]['reason'] == 'timeout'
    assert recorder.steps[0] == (3, rows)
    observed.close()
    observed.close()
    assert recorder.closed == 1
    assert recorder.events == ['reset', 'tick', 'tick', 'tick', 'step', 'close']


def test_observer_sees_every_control_interval_tick_including_late_contact_veto():
    backend = make(PressTask(num_envs=1, wrong_at=4))
    recorder = ReadOnlyRecorder()
    backend.add_observer(recorder)
    reset_all(backend)
    row, = backend.step(actions(ids=(0,), horizon=2))
    assert len(recorder.ticks) == 4
    assert all(tick['outcomes'][0]['success'] for tick in recorder.ticks[:3])
    assert recorder.ticks[-1]['outcomes'][0]['info']['wrong_contact']
    assert not recorder.ticks[-1]['outcomes'][0]['success']
    assert recorder.steps[0][1] == [row]
    assert backend.engine.world.count == 4


def test_fused_observer_step_precedes_reset_and_preserves_old_terminal_state():
    task = LiftTask()
    task.thresholds[0] = .01
    backend = make(task)
    recorder = ReadOnlyRecorder()
    backend.add_observer(recorder)
    reset_all(backend)
    reply = backend.step_and_reset(actions(horizon=1), reset_seed=17, reset_options={}, max_active_envs=2)
    assert recorder.events == ['reset', 'tick', 'tick', 'tick', 'tick', 'step', 'reset']
    assert recorder.steps[0][1][0]['observation']['state'][0] == pytest.approx(.5)
    assert recorder.resets[-1][2][0]['observation']['state'] == [0., 0.]
    assert recorder.steps[0][1][0]['info']['physics_steps'] == 1
    assert reply['items'][0]['observation']['state'][0] == pytest.approx(.5)
    assert backend.engine.world.count == 4


@pytest.mark.parametrize('callback', ['on_reset', 'after_tick', 'after_step'])
def test_observer_cannot_advance_physical_time_and_continue(callback):
    backend = make()
    observer = SimpleNamespace()
    def illicit_step(backend, *args):
        backend.engine.tick([0, 1])
    setattr(observer, callback, illicit_step)
    if callback != 'on_reset':
        reset_all(backend)
    backend.add_observer(observer)
    operation = (lambda: reset_all(backend)) if callback == 'on_reset' else (
        lambda: backend.step(actions(horizon=1)))
    with pytest.raises(RuntimeError, match='observer advanced physical time'):
        operation()
    count = backend.engine.world.count
    with pytest.raises(RuntimeError, match='faulted'):
        backend.step(actions(horizon=1))
    assert backend.engine.world.count == count


def test_duplicate_observer_registration_is_rejected_without_duplicate_callbacks():
    backend = make()
    recorder = ReadOnlyRecorder()
    backend.add_observer(recorder)
    with pytest.raises(ValueError, match='already attached'):
        backend.add_observer(recorder)
    reset_all(backend)
    backend.close()
    assert len(recorder.resets) == recorder.closed == 1
