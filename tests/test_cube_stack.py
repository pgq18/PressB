"""Cube stacking contracts using measured geometry, without Kit or hardware.

Outcome trajectories below are independent geometric fixtures: an actual lift
with simultaneous bilateral finger forces, followed by released support. They
exercise the production detector, not a second implementation of its rules.
"""
from copy import deepcopy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.tasks.cube_stack.task import CubeStackOutcome


EDGE = .03
TABLE = .76


def builtins_only(value):
    """JSON accepts np.float64 on some versions; require plain wire values too."""
    if isinstance(value, dict):
        assert all(type(k) in (str, int) for k in value)
        for item in value.values():
            builtins_only(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            builtins_only(item)
    else:
        assert value is None or type(value) in (str, bool, int, float), type(value)
        if type(value) is float:
            assert math.isfinite(value)


def measured(*, seconds=0., stacked=True):
    red = [.3, 0., TABLE + .045] if stacked else [.2, 0., TABLE + .07]
    return dict(seconds=seconds,
        positions=np.array([red, [.3, 0., TABLE + .015]], dtype=np.float64),
        quaternions_wxyz=np.tile([1., 0., 0., 0.], (2, 1)),
        velocities=np.zeros((2, 6)), gripper_width_m=.05 if stacked else .028,
        contacts=dict(red_finger_forces_n=[0., 0.] if stacked else [.1, .1],
            blue_finger_forces_n=[0., 0.], red_blue_support_force_n=.3 if stacked else 0.,
            blue_table_support_force_n=.6, red_table_force_n=0.))


def tracker(**config):
    return CubeStackOutcome(EDGE, EDGE, TABLE, **config)


def prime_lift(outcome):
    result = outcome.update(**measured(stacked=False))
    assert result['grasped_and_lifted'] and not result['success']
    return result


def hold(outcome, change=lambda frame: None, *, start=.1, stop=.7):
    results = []
    for seconds in np.linspace(start, stop, 7):
        frame = measured(seconds=float(seconds))
        change(frame)
        results.append(outcome.update(**frame))
    return results


def test_physical_lift_then_released_stable_stack_succeeds_after_dwell_only_once():
    outcome = tracker()
    prime_lift(outcome)
    result = hold(outcome)
    assert not result[0]['success']
    assert result[-1]['success']
    assert sum(row['success_event'] for row in result) == 1
    assert result[-1]['stable_seconds'] >= .5
    assert result[-1]['red_bottom_world_m'] == pytest.approx(TABLE + EDGE)
    assert result[-1]['blue_top_world_m'] == pytest.approx(TABLE + EDGE)
    assert result[-1]['blue_bottom_world_m'] == pytest.approx(TABLE)
    assert result[-1]['xy_overlap_fraction'] == pytest.approx(1.)
    builtins_only(result)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('history', ['none', 'single_finger', 'separate_fingers', 'lift_without_grasp', 'grasp_without_lift'])
def test_stack_requires_simultaneous_bilateral_grasp_during_measured_lift(history):
    outcome = tracker()
    if history != 'none':
        frame = measured(stacked=False)
        if history == 'single_finger':
            frame['contacts']['red_finger_forces_n'] = [.1, 0.]
        elif history == 'separate_fingers':
            frame['contacts']['red_finger_forces_n'] = [.1, 0.]
            outcome.update(**frame)
            frame['seconds'] = .01
            frame['contacts']['red_finger_forces_n'] = [0., .1]
        elif history == 'lift_without_grasp':
            frame['contacts']['red_finger_forces_n'] = [0., 0.]
        elif history == 'grasp_without_lift':
            frame['positions'][0, 2] = TABLE + EDGE/2
        outcome.update(**frame)
    rows = hold(outcome)
    assert all(not row['success'] and not row['grasped_and_lifted'] for row in rows)


@pytest.mark.parametrize('support', ['blue', 'table'])
def test_touching_a_supported_red_cube_does_not_establish_grasp_and_lift_history(support):
    outcome = tracker()
    frame = measured(stacked=True)
    frame['gripper_width_m'] = EDGE
    frame['contacts']['red_finger_forces_n'] = [.1, .1]
    if support == 'table':
        frame['contacts']['red_table_force_n'] = .2
        frame['contacts']['red_blue_support_force_n'] = 0.
    result = outcome.update(**frame)
    assert not result['grasped_and_lifted']
    assert all(not row['success'] for row in hold(outcome))


def invalid_stack(frame, condition):
    if condition == 'side_contact':
        frame['positions'][0, 0] += EDGE
        frame['positions'][0, 2] = TABLE + EDGE/2
    elif condition == 'tabletop':
        frame['positions'][0] = [.2, 0., TABLE + EDGE/2]
        frame['contacts']['red_table_force_n'] = .2
    elif condition == 'floating_red':
        frame['positions'][0, 2] += .01
    elif condition == 'interpenetrating':
        frame['positions'][0, 2] -= .01
    elif condition == 'floating_blue':
        frame['positions'][:, 2] += .01
    elif condition == 'blue_below_table':
        frame['positions'][:, 2] -= .01
    elif condition in ('red_tilted', 'blue_tilted'):
        i = 0 if condition == 'red_tilted' else 1
        angle = math.radians(25)
        frame['quaternions_wxyz'][i] = [math.cos(angle/2), math.sin(angle/2), 0., 0.]
    elif condition in ('red_inverted', 'blue_inverted'):
        frame['quaternions_wxyz'][0 if condition == 'red_inverted' else 1] = [0., 1., 0., 0.]
    elif condition == 'insufficient_overlap':
        frame['positions'][0, 0] += EDGE*.6
    elif condition == 'still_closed':
        frame['gripper_width_m'] = .025
    elif condition == 'red_still_held':
        frame['contacts']['red_finger_forces_n'] = [.1, .1]
    elif condition == 'blue_still_held':
        frame['contacts']['blue_finger_forces_n'] = [.1, .1]
    elif condition == 'missing_block_support':
        frame['contacts']['red_blue_support_force_n'] = 0.
    elif condition == 'missing_table_support':
        frame['contacts']['blue_table_support_force_n'] = 0.
    elif condition == 'weak_block_support':
        frame['contacts']['red_blue_support_force_n'] = .001
    elif condition == 'red_also_on_table':
        frame['contacts']['red_table_force_n'] = .2
    elif condition == 'moving_red':
        frame['velocities'][0, 0] = .1
    elif condition == 'moving_blue':
        frame['velocities'][1, 0] = .1
    elif condition == 'spinning_red':
        frame['velocities'][0, 5] = 1.
    elif condition == 'spinning_blue':
        frame['velocities'][1, 5] = 1.
    else:
        raise AssertionError(condition)


@pytest.mark.parametrize('condition', ['side_contact', 'tabletop', 'floating_red', 'interpenetrating',
    'floating_blue', 'blue_below_table', 'red_tilted', 'blue_tilted', 'red_inverted', 'blue_inverted',
    'insufficient_overlap', 'still_closed', 'red_still_held', 'blue_still_held',
    'missing_block_support', 'missing_table_support', 'weak_block_support', 'red_also_on_table',
    'moving_red', 'moving_blue', 'spinning_red', 'spinning_blue'])
def test_invalid_measured_stack_never_accumulates_success(condition):
    outcome = tracker()
    prime_lift(outcome)
    results = hold(outcome, lambda frame: invalid_stack(frame, condition))
    assert all(not row['success'] and not row['success_event'] for row in results)
    assert results[-1]['stable_seconds'] == 0.
    builtins_only(results)


@pytest.mark.parametrize('break_condition', ['missing_block_support', 'red_still_held', 'moving_blue', 'floating_red'])
def test_stability_duration_must_be_continuous(break_condition):
    outcome = tracker()
    prime_lift(outcome)
    hold(outcome, start=.1, stop=.4)
    bad = measured(seconds=.45)
    invalid_stack(bad, break_condition)
    interrupted = outcome.update(**bad)
    assert not interrupted['success'] and interrupted['stable_seconds'] == 0.
    results = hold(outcome, start=.5, stop=.9)
    assert all(not row['success'] for row in results)
    assert outcome.update(**measured(seconds=1.01))['success']


@pytest.mark.parametrize('yaw_degrees', [-170., -45., 0., 45., 179.])
def test_shared_arbitrary_yaw_and_quaternion_sign_preserve_upright_stack(yaw_degrees):
    outcomes = []
    for sign in [1., -1.]:
        outcome = tracker()
        prime_lift(outcome)
        angle = math.radians(yaw_degrees)/2
        def yaw(frame):
            frame['quaternions_wxyz'][:] = np.array([math.cos(angle), 0., 0., math.sin(angle)])*sign
        outcomes.append(hold(outcome, yaw)[-1])
    assert outcomes[0] == outcomes[1]
    assert outcomes[0]['success']


@pytest.mark.parametrize('field,change', [
    ('positions', np.zeros((1, 3))), ('positions', np.full((2, 3), np.nan)),
    ('quaternions_wxyz', np.zeros((2, 4))), ('quaternions_wxyz', np.ones((2, 3))),
    ('velocities', np.zeros((2, 3))), ('velocities', np.full((2, 6), np.inf)),
    ('gripper_width_m', float('nan')), ('seconds', float('nan')),
])
def test_invalid_physical_samples_are_rejected(field, change):
    frame = measured()
    frame[field] = change
    with pytest.raises((ValueError, TypeError)):
        tracker().update(**frame)


@pytest.mark.parametrize('option,value', [('stable_seconds', -1.), ('min_lift_height_m', -1.),
    ('min_xy_overlap_fraction', 1.1), ('max_tilt_deg', 100.),
    ('support_height_tolerance_m', -1.), ('max_linear_speed_m_s', 0.),
    ('min_grasp_force_n', float('nan')), ('unknown_option', 1.)])
def test_invalid_outcome_configuration_is_rejected(option, value):
    with pytest.raises((ValueError, TypeError)):
        tracker(**{option: value})


def cube_config(**changes):
    from pressb.tasks.cube_stack.configuration import load_config
    return load_config(changes)


def test_task_geometry_is_added_without_changing_playground_calibration():
    from pressb.tasks.playground.backend import load_config as workspace_config
    original = workspace_config({'scene_config': 'configs/calibration/playground_20261109_global_refined.json'})
    cfg = cube_config()
    for key in ('table_height', 'table_width', 'table_depth', 'table_x_min',
                'robot_base_xy', 'robot_base_height_above_table_m', 'initial_state_joint_rad_m',
                'wrist_camera', 'wrist_camera_resolution', 'up_camera', 'lighting', 'materials'):
        assert cfg['scene'][key] == original['scene'][key], key
    assert len(cfg['scene']['objects']) == 2
    objects = {obj['name']: obj for obj in cfg['scene']['objects']}
    assert set(objects) == {'red', 'blue'}
    for color, obj in objects.items():
        assert obj['size'] == [cfg['cubes'][color]['edge_m']]*3
        assert obj['mass'] == cfg['cubes'][color]['mass_kg']


def test_sampling_is_reproducible_per_environment_and_independent_of_other_calls():
    from pressb.tasks.cube_stack.configuration import sample_objects
    cfg = cube_config(randomization={'yaw_range_deg': [-45., 45.]})
    expected = {i: sample_objects(cfg, i, seed) for i, seed in [(0, 100), (1, 101), (2, 102)]}
    for i in range(40):
        sample_objects(cfg, 17, i)
    actual = {i: sample_objects(cfg, i, seed) for i, seed in [(2, 102), (0, 100), (1, 101)]}
    assert actual == expected
    assert len({json.dumps(value, sort_keys=True) for value in expected.values()}) == 3
    builtins_only(actual)
    json.dumps(actual, allow_nan=False)


def test_randomized_rotated_cubes_stay_inside_bounds_and_do_not_overlap():
    from pressb.tasks.cube_stack.configuration import sample_objects
    cfg = cube_config(randomization={'yaw_range_deg': [-180., 180.]})
    bounds = cfg['randomization']
    for seed in range(160):
        objects = sample_objects(cfg, seed % 8, seed)
        positions = []
        for color, obj in objects.items():
            xy = np.array(obj['center_xy'])
            edge = cfg['cubes'][color]['edge_m']
            angle = math.radians(obj['yaw_deg'])
            # Exact axis extent of a square, independently checked against the
            # advertised rectangle instead of calling production validation.
            extent = edge/2*(abs(math.cos(angle))+abs(math.sin(angle)))
            for dim, key in enumerate(('x_range_m', 'y_range_m')):
                assert xy[dim]-extent >= bounds[key][0]-1e-12
                assert xy[dim]+extent <= bounds[key][1]+1e-12
            positions.append(xy)
        distance = np.linalg.norm(positions[0]-positions[1])
        assert distance >= bounds['min_center_distance_m']-1e-12
        # Default minimum separation exceeds the two circumradii, proving no
        # polygon overlap without reusing the sampler's polygon clipping code.
        assert distance > sum(c['edge_m']/math.sqrt(2) for c in cfg['cubes'].values())


def test_explicit_layout_and_disabled_randomization_preserve_input_values():
    from pressb.tasks.cube_stack.configuration import sample_objects
    source = {'randomization': {'enabled': False}}
    original = deepcopy(source)
    cfg = cube_config(**source)
    for seed in (0, 1, 2**63-1):
        sampled = sample_objects(cfg, 0, seed)
        for color, value in sampled.items():
            assert value['center_xy'] == cfg['cubes'][color]['center_xy']
            assert value['yaw_deg'] == cfg['cubes'][color]['yaw_deg']
    overrides = {'red': {'center_xy': [.29, -.08], 'yaw_deg': 45.},
                 'blue': {'center_xy': [.4, .08], 'yaw_deg': -30.}}
    preserved = deepcopy(overrides)
    randomized = cube_config()
    sampled = sample_objects(randomized, 2, 51, overrides)
    for color, spec in overrides.items():
        assert sampled[color]['center_xy'] == spec['center_xy']
        assert sampled[color]['yaw_deg'] == spec['yaw_deg']
    assert source == original and overrides == preserved


@pytest.mark.parametrize('overrides', [
    {'red': {'center_xy': [.3, 0.]}, 'blue': {'center_xy': [.3, 0.]}},
    {'red': {'center_xy': [.25, 0.]}, 'blue': {'center_xy': [.4, 0.]}},
    {'red': {'center_xy': [float('nan'), 0.]}},
    {'red': {'yaw_deg': True}}, {'green': {}}, {'red': {'mass': .1}},
])
def test_invalid_or_overlapping_explicit_layout_is_rejected(overrides):
    from pressb.tasks.cube_stack.configuration import sample_objects
    with pytest.raises(ValueError):
        sample_objects(cube_config(), 0, 12, overrides)


@pytest.mark.parametrize('change', [
    {'randomization': {'enabled': 'yes'}}, {'randomization': {'x_range_m': [.4, .3]}},
    {'randomization': {'max_attempts': True}}, {'randomization': {'max_attempts': 0}},
    {'randomization': {'min_center_distance_m': -1}},
    {'randomization': {'x_range_m': [.3, .31]}},
    {'cubes': {'red': {'edge_m': 0.}}}, {'cubes': {'blue': {'mass_kg': -.1}}},
    {'cubes': {'red': {'edge_m': .2}}}, {'cubes': {'red': {'color': [1., 2., 0.]}}},
    {'cubes': {'yellow': {}}}, {'terminate_on_success': 1},
    {'reward': {'success': float('nan')}}, {'reward': {'failure': True}},
    {'scene_overrides': {'objects': []}}, {'unknown': 1},
])
def test_invalid_cube_configuration_is_rejected_before_simulator_initialization(change):
    with pytest.raises((ValueError, TypeError)):
        cube_config(**change)


@pytest.mark.parametrize('seed', [True, -1, 2**63, .5])
def test_invalid_sampling_seed_is_rejected(seed):
    from pressb.tasks.cube_stack.configuration import sample_objects
    with pytest.raises(ValueError):
        sample_objects(cube_config(), 0, seed)


class MeasuredBodies:
    def __init__(self, count):
        self.positions = np.zeros((count, 3))
        self.rotations = np.tile([1., 0., 0., 0.], (count, 1))
        self.velocities = np.ones((count, 6))
        self.pose_writes, self.velocity_writes = [], []

    def get_world_poses(self):
        return self.positions.copy(), self.rotations.copy()

    def get_velocities(self):
        return self.velocities.copy()

    def set_world_poses(self, positions, rotations, rows):
        self.pose_writes.append(list(rows))
        self.positions[rows], self.rotations[rows] = positions, rotations

    def set_velocities(self, values, rows):
        self.velocity_writes.append(list(rows))
        self.velocities[rows] = values


def bound_task(**changes):
    from pressb.tasks.cube_stack.parallel import CubeStackTask
    config = dict(num_envs=2, randomization={'enabled': False},
        cubes={'red': {'edge_m': EDGE, 'yaw_deg': 0.}, 'blue': {'edge_m': EDGE, 'yaw_deg': 0.}})
    config.update(changes)
    task = CubeStackTask(config)
    offsets = np.array([[0., 0., 0.], [6., -6., .1]])
    rows = np.array([[3, 1], [0, 2]])  # PhysX view order is deliberately scrambled.
    bodies = MeasuredBodies(4)
    scene = {'root_path': '/World/CubeStack', 'table_top_z': TABLE, 'objects': {}}
    for j, color in enumerate(('red', 'blue')):
        cube = task.config['cubes'][color]
        scene['objects'][color] = {'path': f'/World/CubeStack/Objects/{color}'}
        for eid in range(2):
            bodies.positions[rows[eid, j]] = np.array([*cube['center_xy'], TABLE+EDGE/2])+offsets[eid]
    widths = np.full(2, .05)
    engine = SimpleNamespace(num_envs=2, offsets=offsets, world=SimpleNamespace(current_time=4.),
        template=SimpleNamespace(data={'scene': scene}, robot_path='/World/Piper'),
        rigid_bodies=lambda paths: (bodies, rows),
        backend=SimpleNamespace(ready=np.ones(2, dtype=bool), ticks=np.array([0, 0]), reset_specs=[None]*2),
        robot=SimpleNamespace(observations=lambda ids: {i: {'state': [0.]*6+[float(widths[i])]} for i in ids}),
        widths=widths)
    task.bind(engine)
    task.after_settle()
    specs = {i: task.sample_reset(i, 12, {}) for i in range(2)}
    engine.backend.reset_specs[:] = [deepcopy(specs[i]) for i in range(2)]
    task.reset_task([0, 1], specs)
    bodies.pose_writes.clear()
    bodies.velocity_writes.clear()
    return task


def emit_contact(task, eid, first, second, *, force=.3, normal=(0., 0., 1.),
                 separation=0., impulse_direction=(0., 0., 1.), collider_only=False):
    impulse = np.asarray(impulse_direction)*force*task.execution['physics_dt']
    paths = [first, second, first+'/collision', second+'/collision']
    if collider_only:
        paths[:2] = ['/World/Unknown0', '/World/Unknown1']
    task.on_contact(eid, paths, [SimpleNamespace(impulse=impulse, normal=normal, separation=separation)])


def test_indexed_cube_reset_preserves_peer_pose_velocity_and_outcome_history():
    task = bound_task()
    task.bodies.velocities[:] = 1.
    task.contact_counts[:] = 7
    task.contacts[0]['red_blue_support_force_n'] = .7
    task.checkers[0].grasped_and_lifted = True
    task.last_outcomes[0] = {'fixture': 'unchanged'}
    peer = deepcopy(task.diagnostic_state([0])[0])
    peer_checker, peer_contacts = task.checkers[0], deepcopy(task.contacts[0])
    spec = task.sample_reset(1, 33, {'objects': {
        'red': {'center_xy': [.28, -.1], 'yaw_deg': 45.},
        'blue': {'center_xy': [.4, .1], 'yaw_deg': -30.}}})
    output = task.reset_task([1], {1: spec})
    assert task.diagnostic_state([0])[0] == peer
    assert task.checkers[0] is peer_checker and task.checkers[0].grasped_and_lifted
    assert task.contacts[0] == peer_contacts and task.last_outcomes[0] == {'fixture': 'unchanged'}
    assert np.all(task.contact_counts[0] == 7) and not task.contact_counts[1].any()
    assert not task.checkers[1].grasped_and_lifted and task.last_outcomes[1] == {}
    assert task.bodies.pose_writes == task.bodies.velocity_writes == [task.rows[1].tolist()]
    for j, color in enumerate(('red', 'blue')):
        row = task.rows[1, j]
        np.testing.assert_allclose(task.bodies.positions[row, :2],
            np.array(spec['objects'][color]['center_xy'])+task.engine.offsets[1, :2])
        assert task.bodies.positions[row, 2] == pytest.approx(TABLE+EDGE/2+.1)
        np.testing.assert_array_equal(task.bodies.velocities[row], np.zeros(6))
    assert task.checkers[1].tabletop_z == pytest.approx(TABLE+.1)
    assert task.engine.world.current_time == 4. and not task.engine.backend.ticks.any()
    assert output[1]['task_outcome_available'] is True
    builtins_only(output)
    json.dumps(output, allow_nan=False)


@pytest.mark.parametrize('normal,separation,direction,accepted', [
    ((0, 0, 1), 0., (0, 0, 1), True), ((0, 0, -1), 0., (0, 0, -1), True),
    ((1, 0, 0), 0., (0, 0, 1), False), ((0, 0, 1), .02, (0, 0, 1), False),
    ((0, 0, 1), -.02, (0, 0, 1), False), ((0, 0, 1), 0., (1, 0, 0), False),
])
def test_only_near_surface_vertical_contact_impulses_count_as_support(normal, separation, direction, accepted):
    task = bound_task()
    emit_contact(task, 1, task.object_paths['red'], task.object_paths['blue'], force=.3,
        normal=normal, separation=separation, impulse_direction=direction, collider_only=True)
    assert task.contacts[1]['red_blue_support_force_n'] == pytest.approx(.3 if accepted else 0.)
    assert task.contacts[0]['red_blue_support_force_n'] == 0.
    assert not task.contact_counts.any()


def test_finger_and_table_contacts_are_classified_by_exact_object_and_live_slot():
    task = bound_task()
    red, blue = task.object_paths['red'], task.object_paths['blue']
    emit_contact(task, 0, red, '/World/Piper/link7', force=.1, collider_only=True)
    emit_contact(task, 0, red, '/World/Piper/link8', force=.2)
    emit_contact(task, 0, blue, task.table_path, force=.4)
    emit_contact(task, 0, red+'_other', '/World/Piper/link7', force=3.)
    emit_contact(task, 0, red, '/World/Piper/link6', force=4.)
    np.testing.assert_allclose(task.contacts[0]['red_finger_forces_n'], [.1, .2])
    assert task.contacts[0]['blue_table_support_force_n'] == pytest.approx(.4)
    assert task.contact_counts[0].tolist() == [[1, 1], [0, 0]]
    task.engine.backend.ready[1] = False
    emit_contact(task, 1, red, '/World/Piper/link7', force=9.)
    assert task.contacts[1] == task._empty_contacts()
    counts = task.contact_counts.copy()
    task.before_tick([0])
    assert task.contacts[0] == task._empty_contacts()
    np.testing.assert_array_equal(task.contact_counts, counts)


@pytest.mark.parametrize('field,value', [('impulse', [np.nan, 0., 0.]),
    ('normal', [0., np.inf, 1.]), ('separation', np.nan), ('impulse', [1., 2.])])
def test_invalid_measured_contact_is_rejected(field, value):
    task = bound_task()
    values = dict(impulse=[0., 0., .001], normal=[0., 0., 1.], separation=0.)
    values[field] = value
    red, blue = task.object_paths['red'], task.object_paths['blue']
    with pytest.raises(RuntimeError, match='contact'):
        task.on_contact(0, [red, blue, red+'/collision', blue+'/collision'], [SimpleNamespace(**values)])


def feed_task(task, frame, *, eid=0):
    task.bodies.positions[task.rows[eid]] = frame['positions']+task.engine.offsets[eid]
    task.bodies.rotations[task.rows[eid]] = frame['quaternions_wxyz']
    task.bodies.velocities[task.rows[eid]] = frame['velocities']
    task.engine.widths[eid] = frame['gripper_width_m']
    task.contacts[eid] = deepcopy(frame['contacts'])
    task.engine.backend.ticks[eid] = round(frame['seconds']/task.execution['physics_dt'])
    return task.after_tick([eid])[eid]


@pytest.mark.parametrize('terminate', [True, False])
def test_task_reward_is_a_single_physical_success_event_and_json_serializable(terminate):
    task = bound_task(terminate_on_success=terminate, reward={'success': 2., 'failure': -.5})
    feed_task(task, measured(seconds=0., stacked=False), eid=1)
    result = [feed_task(task, measured(seconds=float(t)), eid=1) for t in np.linspace(.1, .7, 7)]
    assert result[-1]['success'] and result[-1]['terminated'] is terminate
    assert sum(row['reward'] for row in result) == 2.
    assert all(not row['truncated'] for row in result)
    builtins_only(result)
    json.dumps(result, allow_nan=False)
    assert task.execution['terminal_boundary'] == 'physics_tick'


def test_continuing_recording_separates_current_collapsed_stack_from_ever_success():
    task = bound_task(terminate_on_success=False)
    feed_task(task, measured(seconds=0., stacked=False))
    for seconds in np.linspace(.1, .7, 7):
        row = feed_task(task, measured(seconds=float(seconds)))
    assert row['success'] and row['info']['physical_outcome']['ever_success']
    collapsed = measured(seconds=.8)
    invalid_stack(collapsed, 'tabletop')
    row = feed_task(task, collapsed)
    assert not row['success'] and not row['terminated']
    assert row['info']['physical_outcome']['ever_success'] and row['reward'] == 0.
    first = feed_task(task, measured(seconds=.9))
    assert not first['success'] and first['info']['physical_outcome']['ever_success']
    assert not feed_task(task, measured(seconds=1.3))['success']
    recovered = feed_task(task, measured(seconds=1.41))
    assert recovered['success'] and recovered['reward'] == 0.
    assert not recovered['info']['physical_outcome']['success_event']


def test_cube_task_metadata_and_reset_instruction_describe_stack_not_empty_workspace():
    from pressb.runtime.contract import validate_metadata
    task = bound_task()
    metadata = validate_metadata(task.metadata(), 'simulation')
    assert metadata['task_id'] == 'cube_stack'
    assert metadata['observation']['cameras'] == ['up', 'wrist']
    assert metadata['action']['names'] == [f'joint{i}' for i in range(1, 7)]+['gripper_width']
    assert metadata['action']['units'] == ['rad']*6+['m']
    assert metadata['reward_schema']['kind'] == 'sparse_physical_event'
    assert metadata['task_instruction'] == task.sample_reset(0, 4, {})['task']
    assert metadata['task_instruction'] == task.sample_reset(1, 88, {})['task']
    with pytest.raises(ValueError, match='instruction'):
        task.sample_reset(0, 4, {'task': 'Put the blue cube on the red cube.'})
    builtins_only(metadata)
    json.dumps(metadata, allow_nan=False)
