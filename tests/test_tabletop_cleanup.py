"""Independent measured-state and mask tests for two-object tabletop cleanup.

CPU fixtures describe explicit poses, hull vertices and contacts; no simulator,
physical robot or existing training endpoint is used.
"""
from copy import deepcopy
from itertools import product
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.tasks.tabletop_cleanup.task import TabletopCleanupOutcome


KINDS = ('gamepad', 'speaker', 'lidar')
TABLE = .76


def cleanup_config(**changes):
    from pressb.tasks.tabletop_cleanup.configuration import load_config
    return load_config(dict(num_envs=2, **changes))


def actual_footprint(spec, pose):
    """Independent full-surface projection, without calling sampler helpers."""
    from scipy.spatial import ConvexHull
    angle = math.radians(pose['yaw_deg'])
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    points = np.asarray(spec['support_vertices'])[:, :2] @ rotation.T + pose['center_xy']
    return points[ConvexHull(points).vertices]


def polygon_distance(a, b):
    """Distance oracle: edge intersection/containment, then segment distances."""
    from scipy.spatial import ConvexHull
    for vertices, other in ((a, b), (b, a)):
        equations = ConvexHull(vertices).equations
        if np.any(np.all(other @ equations[:, :2].T + equations[:, 2] <= 1e-12, axis=1)):
            return 0.
    best = math.inf
    for p, q in zip(a, np.roll(a, -1, axis=0)):
        for r, s in zip(b, np.roll(b, -1, axis=0)):
            vectors = np.column_stack((q-p, r-s))
            if abs(np.linalg.det(vectors)) > 1e-12:
                parameters = np.linalg.solve(vectors, r-p)
                if np.all(parameters >= 0.) and np.all(parameters <= 1.):
                    return 0.
            for point, start, end in ((p, r, s), (q, r, s), (r, p, q), (s, p, q)):
                delta = end-start
                projection = start+np.clip((point-start) @ delta/(delta @ delta), 0., 1.)*delta
                best = min(best, float(np.linalg.norm(point-projection)))
    return best


def builtin_values(value):
    if isinstance(value, dict):
        assert all(type(key) in (str, int) for key in value)
        for item in value.values():
            builtin_values(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            builtin_values(item)
    else:
        assert value is None or type(value) in (str, bool, int, float), type(value)
        if type(value) is float:
            assert math.isfinite(value)


def assets():
    result = []
    for kind, size in zip(KINDS, ([.06, .025, .02], [.025, .025, .035], [.03, .03, .025])):
        vertices = (np.array(list(product((-1., 1.), repeat=3)))*np.asarray(size)/2).tolist()
        result.append(dict(kind=kind, size=list(size), support_vertices=vertices, hull_vertices=vertices,
            mass_kg=.025, grasp_candidates=[dict(center_local_m=[0., 0., 0.],
                closing_axis_local=[0., 1., 0.], opening_m=size[1])]))
    return result


def bin_spec(*, yaw=0., offset=(0., 0., 0.)):
    # The black tray has 222 x 148 x 49 mm external dimensions and a 3 mm base.
    return dict(center_world=(np.array([.36, .14, TABLE+.049/2])+offset).tolist(),
        yaw_deg=yaw, inner_min_local=[-.108, -.071, -.049/2+.003],
        inner_max_local=[.108, .071, .049/2])


def outcome(*, mask=(True, True, False), tray=None, specs=None, **options):
    return TabletopCleanupOutcome(assets() if specs is None else specs, TABLE,
        bin_spec() if tray is None else tray, list(mask), **options)


def contact_zeros():
    return dict(finger_forces_n=np.zeros((3, 2)), table_forces_n=np.zeros(3),
        bin_floor_forces_n=np.zeros(3), object_support_forces_n=np.zeros((3, 3)))


def in_bin(*, seconds=0., mask=(True, True, False), tray=None, specs=None):
    specs, tray = assets() if specs is None else specs, bin_spec() if tray is None else tray
    yaw = math.radians(tray['yaw_deg'])
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    positions = np.full((3, 3), -20.)  # Unselected pooled bodies are outside the task.
    quaternions = np.tile([math.cos(yaw/2), 0., 0., math.sin(yaw/2)], (3, 1))
    contacts = contact_zeros()
    for column, index in enumerate(np.flatnonzero(mask)):
        local_xy = np.array([-.045 if column == 0 else .045, 0.])
        positions[index, :2] = rotation@local_xy+tray['center_world'][:2]
        positions[index, 2] = tray['center_world'][2]+tray['inner_min_local'][2]+specs[index]['size'][2]/2
        contacts['bin_floor_forces_n'][index] = .4
    return dict(seconds=seconds, positions=positions, quaternions_wxyz=quaternions,
        velocities=np.zeros((3, 6)), gripper_width_m=.08, contacts=contacts)


def prime_grasps(detector, *, mask=(True, True, False), tray=None, specs=None):
    """Move each selected item through unsupported bilateral lift sequentially."""
    for step, index in enumerate(np.flatnonzero(mask)):
        frame = in_bin(seconds=step*.01, mask=mask, tray=tray, specs=specs)
        frame['positions'][index] = [.25, -.1, TABLE+.09]
        frame['gripper_width_m'] = .02
        frame['contacts']['bin_floor_forces_n'][index] = 0.
        frame['contacts']['finger_forces_n'][index] = [.1, .1]
        detector.update(**frame)


def dwell(detector, change=lambda frame: None, *, mask=(True, True, False), tray=None,
          specs=None, start=.1, stop=.7):
    result = []
    for seconds in np.linspace(start, stop, 7):
        frame = in_bin(seconds=float(seconds), mask=mask, tray=tray, specs=specs)
        change(frame)
        result.append(detector.update(**frame))
    return result


@pytest.mark.parametrize('mask', [(True, True, False), (True, False, True), (False, True, True)])
def test_both_selected_objects_need_grasp_then_complete_supported_released_dwell(mask):
    detector = outcome(mask=mask)
    prime_grasps(detector, mask=mask)
    rows = dwell(detector, mask=mask)
    assert not rows[0]['success'] and rows[-1]['success']
    assert sum(row['success_event'] for row in rows) == 1
    builtin_values(rows)
    json.dumps(rows, allow_nan=False)


@pytest.mark.parametrize('history', ['none', 'one_object', 'one_finger', 'separate_fingers', 'supported_touch'])
def test_partial_or_fake_grasp_history_cannot_complete_the_two_object_task(history):
    detector = outcome()
    if history != 'none':
        for step, index in enumerate((0, 1)):
            if history == 'one_object' and index == 1:
                continue
            frame = in_bin(seconds=step*.02)
            if history != 'supported_touch':
                frame['positions'][index, 2] = TABLE+.09
                frame['contacts']['bin_floor_forces_n'][index] = 0.
            frame['gripper_width_m'] = .02
            frame['contacts']['finger_forces_n'][index] = [.1, 0.] if history in ('one_finger', 'separate_fingers') else [.1, .1]
            detector.update(**frame)
            if history == 'separate_fingers':
                frame['seconds'] += .01
                frame['contacts']['finger_forces_n'][index] = [0., .1]
                detector.update(**frame)
    assert all(not row['success'] for row in dwell(detector))


def invalid(frame, condition):
    contacts = frame['contacts']
    if condition == 'one_left_outside':
        frame['positions'][1] = [.3, -.1, TABLE+.035/2]
        contacts['bin_floor_forces_n'][1] = 0.
        contacts['table_forces_n'][1] = .4
    elif condition == 'center_inside_but_edge_outside':
        frame['positions'][0, 0] = .36+.108-.005
    elif condition == 'yaw_corner_outside':
        frame['positions'][0, 1] = .14+.071-.015
        a = math.pi/4
        frame['quaternions_wxyz'][0] = [math.cos(a/2), 0., 0., math.sin(a/2)]
    elif condition == 'floor_penetration':
        frame['positions'][0, 2] -= .01
    elif condition == 'air_without_support':
        frame['positions'][0, 2] += .04
        contacts['bin_floor_forces_n'][0] = 0.
    elif condition == 'air_with_stale_floor_contact':
        frame['positions'][0, 2] += .04
    elif condition == 'still_finger_held':
        contacts['finger_forces_n'][0] = [.1, .1]
    elif condition == 'still_on_table':
        contacts['table_forces_n'][0] = .4
    elif condition == 'missing_floor_support':
        contacts['bin_floor_forces_n'][0] = 0.
    elif condition == 'moving':
        frame['velocities'][1, 0] = .1
    elif condition == 'spinning':
        frame['velocities'][0, 5] = 1.
    elif condition == 'inactive_support':
        contacts['bin_floor_forces_n'][0] = 0.
        contacts['object_support_forces_n'][0, 2] = contacts['object_support_forces_n'][2, 0] = .4
        contacts['bin_floor_forces_n'][2] = 1.
    elif condition == 'unsupported_cycle':
        contacts['bin_floor_forces_n'][:] = 0.
        contacts['object_support_forces_n'][0, 1] = contacts['object_support_forces_n'][1, 0] = .4
    elif condition == 'self_support':
        contacts['bin_floor_forces_n'][0] = 0.
        contacts['object_support_forces_n'][0, 0] = .4
    else:
        raise AssertionError(condition)


@pytest.mark.parametrize('condition', ['one_left_outside', 'center_inside_but_edge_outside', 'yaw_corner_outside',
    'floor_penetration', 'air_without_support', 'air_with_stale_floor_contact', 'still_finger_held',
    'still_on_table', 'missing_floor_support', 'moving', 'spinning', 'inactive_support', 'unsupported_cycle'])
def test_invalid_containment_release_or_support_cannot_succeed(condition):
    detector = outcome()
    prime_grasps(detector)
    assert all(not row['success'] for row in dwell(detector, lambda frame: invalid(frame, condition)))


def test_self_support_is_rejected_as_an_invalid_contact_graph():
    frame = in_bin()
    invalid(frame, 'self_support')
    with pytest.raises(ValueError, match='support itself'):
        outcome().update(**frame)


def test_stable_dwell_restarts_after_interruption_and_reward_event_is_not_repeated():
    detector = outcome()
    prime_grasps(detector)
    assert not dwell(detector, start=.1, stop=.4)[-1]['success']
    interrupted = in_bin(seconds=.45)
    invalid(interrupted, 'moving')
    assert detector.update(**interrupted)['stable_seconds'] == 0.
    assert not dwell(detector, start=.5, stop=.9)[-1]['success']
    assert detector.update(**in_bin(seconds=1.01))['success_event']
    lost = in_bin(seconds=1.1)
    invalid(lost, 'missing_floor_support')
    row = detector.update(**lost)
    assert not row['success'] and row['ever_success']
    recovered = dwell(detector, start=1.2, stop=1.8)
    assert not recovered[0]['success'] and recovered[-1]['success']
    assert not any(row['success_event'] for row in recovered)


def stacked_pair(frame):
    frame['positions'][1, :2] = frame['positions'][0, :2]
    frame['positions'][1, 2] = frame['positions'][0, 2]+.02/2+.035/2
    frame['contacts']['bin_floor_forces_n'][1] = 0.
    frame['contacts']['object_support_forces_n'][0, 1] = .4
    frame['contacts']['object_support_forces_n'][1, 0] = .4


def test_supported_object_stack_may_protrude_rim_but_chain_must_reach_real_bin_floor():
    detector = outcome()
    prime_grasps(detector)
    result = dwell(detector, stacked_pair)
    assert result[-1]['success']
    frame = in_bin(seconds=.8)
    stacked_pair(frame)
    # This physically valid pair has its top above the 49 mm outer box rim.
    assert frame['positions'][1, 2]+.035/2 > TABLE+.049
    frame['contacts']['bin_floor_forces_n'][0] = 0.
    assert not detector.update(**frame)['success']


def test_support_chain_cannot_bridge_a_large_physical_air_gap():
    detector = outcome()
    prime_grasps(detector)
    def floating_chain(frame):
        stacked_pair(frame)
        frame['positions'][1, 2] += .04
    assert all(not row['success'] for row in dwell(detector, floating_chain))


@pytest.mark.parametrize('yaw,offset', [(45., [0., 0., 0.]), (-70., [6., -6., 0.])])
def test_rotated_and_cloned_bin_uses_local_cavity_coordinates(yaw, offset):
    tray = bin_spec(yaw=yaw, offset=offset)
    detector = outcome(tray=tray)
    prime_grasps(detector, tray=tray)
    assert dwell(detector, tray=tray)[-1]['success']


def test_inactive_body_below_table_does_not_fail_or_extend_required_object_set():
    detector = outcome()
    prime_grasps(detector)
    def move_inactive(frame):
        frame['positions'][2] = [1000., 1000., -1000.]
        frame['velocities'][2] = 100.
    assert dwell(detector, move_inactive)[-1]['success']


@pytest.mark.parametrize('mask', [(True, False, False), (True, True, True), (False, False, False), (True, True), (1, 1, 0)])
def test_active_mask_requires_exactly_two_distinct_pool_members(mask):
    with pytest.raises((ValueError, TypeError)):
        outcome(mask=mask)


@pytest.mark.parametrize('field,value', [('positions', np.zeros((2, 3))), ('positions', np.full((3, 3), np.nan)),
    ('velocities', np.full((3, 6), np.inf)), ('quaternions_wxyz', np.zeros((3, 4))),
    ('seconds', float('nan')), ('gripper_width_m', float('nan'))])
def test_invalid_measured_state_is_rejected(field, value):
    frame = in_bin()
    frame[field] = value
    with pytest.raises((ValueError, TypeError)):
        outcome().update(**frame)


def test_sampling_checks_full_visual_geometry_clearance_and_seed_call_independence():
    from pressb.tasks.tabletop_cleanup.configuration import sample_objects
    config = cleanup_config()
    original = deepcopy(config)
    fixed = sample_objects(config, 1, 123)
    seen = set()
    box = config['bin']
    size = np.asarray(box['size_xyz_m'])
    bin_specification = {'support_vertices':
        (np.array(list(product((-1., 1.), repeat=3)))*size/2).tolist()}
    bin_polygon = actual_footprint(bin_specification, box)
    for seed in range(60):
        layout = sample_objects(config, seed % 2, seed)
        names = layout['active_names']
        assert names == [kind for kind in KINDS if kind in names]
        assert len(set(names)) == 2 and set(layout['objects']) == set(names)
        assert layout['active_mask'] == [kind in names for kind in KINDS]
        seen.add(tuple(names))
        polygons = []
        for kind, pose in layout['objects'].items():
            polygon = actual_footprint(config['asset_specs'][KINDS.index(kind)], pose)
            polygons.append(polygon)
            for axis, key in enumerate(('x_range_m', 'y_range_m')):
                low, high = config['randomization'][key]
                assert polygon[:, axis].min() >= low-1e-12
                assert polygon[:, axis].max() <= high+1e-12
            assert polygon_distance(polygon, bin_polygon) >= config['randomization']['clearance_m']-1e-12
        assert polygon_distance(*polygons) >= config['randomization']['clearance_m']-1e-12
        builtin_values(layout)
        json.dumps(layout, allow_nan=False)
    assert seen == {('gamepad', 'speaker'), ('gamepad', 'lidar'), ('speaker', 'lidar')}
    assert sample_objects(config, 1, 123) == fixed
    assert sample_objects(config, 0, 123) != fixed
    assert config == original


def test_explicit_pair_pose_overrides_are_isolated_and_do_not_mutate_inputs():
    from pressb.tasks.tabletop_cleanup.configuration import sample_objects
    config = cleanup_config()
    overrides = {'lidar': {'center_xy': [.3, -.16], 'yaw_deg': -70.},
                 'speaker': {'center_xy': [.4, .08], 'yaw_deg': 70.}}
    original = deepcopy(overrides)
    a, b = (sample_objects(config, 0, seed, overrides) for seed in (0, 99))
    assert a == b and a['active_names'] == ['speaker', 'lidar']
    assert a['active_mask'] == [False, True, True]
    assert a['objects'] == overrides and overrides == original
    a['objects']['speaker']['center_xy'][0] = 999.
    assert sample_objects(config, 0, 0, overrides) == b


@pytest.mark.parametrize('overrides', [
    {}, {'speaker': {}}, {'speaker': {}, 'lidar': {}, 'gamepad': {}},
    {'speaker': {}, 'cube': {}}, {'speaker': {'scale': 2.}, 'lidar': {}},
    {'speaker': {'center_xy': [float('nan'), 0.]}, 'lidar': {}},
    {'speaker': {'yaw_deg': True}, 'lidar': {}},
    {'speaker': {'center_xy': [.3, 0.]}, 'lidar': {'center_xy': [.3, 0.]}},
    {'speaker': {'center_xy': [.9, 0.]}, 'lidar': {'center_xy': [.3, 0.]}},
])
def test_invalid_explicit_object_reset_rejected_without_rng_or_scene_mutation(overrides):
    from pressb.tasks.tabletop_cleanup.configuration import sample_objects
    config = cleanup_config()
    expected = sample_objects(config, 0, 19)
    with pytest.raises(ValueError):
        sample_objects(config, 0, 42, overrides)
    assert sample_objects(config, 0, 19) == expected


@pytest.mark.parametrize('change', [
    {'randomization': {'pool': ['speaker']}}, {'randomization': {'pool': ['speaker', 'speaker']}},
    {'randomization': {'pool': ['speaker', 'cube']}},
    {'randomization': {'fixed_subset': ['speaker', 'speaker']}},
    {'randomization': {'pool': ['speaker', 'lidar'], 'fixed_subset': ['speaker', 'gamepad']}},
    {'randomization': {'x_range_m': [.4, .3]}}, {'randomization': {'y_range_m': [0., 0.]}},
    {'randomization': {'yaw_range_deg': [0., float('inf')]}},
    {'randomization': {'enabled': 1}}, {'randomization': {'max_attempts': 0}},
    {'randomization': {'clearance_m': -.01}}, {'randomization': {'active_count': 3}},
    {'objects': {'speaker': {'scale': 0.}}}, {'objects': {'lidar': {'mass_kg': float('nan')}}},
    {'objects': {'speaker': {'yaw_deg': True}}}, {'objects': {'cube': {}}},
    {'bin': {'size_xyz_m': [.2, -.15, .05]}}, {'bin': {'center_xy': [9., 0.]}},
    {'bin': {'wall_thickness_m': .1}}, {'bin': {'lid': True}},
    {'outcome': {'gripper_release_width_m': 1.}}, {'terminate_on_success': 1},
    {'reward': {'success': float('inf')}}, {'scene_overrides': {'objects': []}},
])
def test_invalid_configuration_rejected_before_isaac_or_robot_initialization(change):
    with pytest.raises((ValueError, TypeError)):
        cleanup_config(**change)


@pytest.mark.parametrize('eid,seed', [(-1, 0), (True, 0), (0, True), (0, -1), (0, 2**63), (0, .1)])
def test_invalid_reset_identity_cannot_silently_change_seed_stream(eid, seed):
    from pressb.tasks.tabletop_cleanup.configuration import sample_objects
    with pytest.raises(ValueError):
        sample_objects(cleanup_config(), eid, seed)


class CleanupBodies:
    """Native tensor-shaped state with explicit row-indexed participation writes."""
    def __init__(self, count):
        self.positions = np.zeros((count, 3))
        self.rotations = np.tile([1., 0., 0., 0.], (count, 1))
        self.velocities = np.zeros((count, 6))
        self.flags = np.zeros((count, 1), dtype=np.uint8)
        self.prim_paths = [f'/fixture/body{row}' for row in range(count)]
        self.pose_writes, self.velocity_writes, self.mask_writes = [], [], []
        self.view = self

    def get_world_poses(self):
        return self.positions.copy(), self.rotations.copy()

    def get_velocities(self):
        return self.velocities.copy()

    def set_world_poses(self, positions, rotations, indices):
        self.pose_writes.append(indices.tolist())
        self.positions[indices], self.rotations[indices] = positions, rotations

    def set_velocities(self, velocities, indices):
        self.velocity_writes.append(indices.tolist())
        self.velocities[indices] = velocities

    def get_disable_simulations(self):
        return self.flags.copy()

    def set_disable_simulations(self, values, indices):
        self.mask_writes.append((indices.tolist(), values[indices].copy()))
        self.flags[indices] = values[indices]


def initialize_cleanup_fixture(task, backend):
    count = task.execution['num_envs']
    assert count == 2
    offsets = np.array([[0., 0., 0.], [6., -6., .1]])
    rows = np.array([[5, 1, 3], [2, 4, 0]])
    bodies, visible = CleanupBodies(6), np.ones((2, 3), bool)
    scene = {'root_path': '/World/TabletopCleanup', 'table_top_z': TABLE,
        'robot_base_position_world': [.015, -.025, TABLE+.014], 'objects': {},
        'bin': {**bin_spec(yaw=task.config['bin']['yaw_deg']), 'floor_path': '/World/TabletopCleanup/Bin/Floor'}}
    for kind in KINDS:
        scene['objects'][kind] = {'path': f'/World/TabletopCleanup/Objects/{kind}'}
    widths = np.full(count, .07)
    engine = SimpleNamespace(num_envs=count, offsets=offsets, world=SimpleNamespace(current_time=4.),
        template=SimpleNamespace(data={'scene': scene}, robot_path='/World/Piper'),
        rigid_bodies=lambda paths: (bodies, rows), backend=backend,
        robot=SimpleNamespace(observations=lambda ids: {i: {'state': [0.]*6+[float(widths[i])]} for i in ids}),
        widths=widths, visible=visible)
    def visibility(eid, index, visible=None):
        if visible is not None:
            engine.visible[eid, index] = visible
        return bool(engine.visible[eid, index])
    task._visibility = visibility
    task.bind(engine)
    task.after_settle()
    return engine


def bound_cleanup(**changes):
    from pressb.tasks.tabletop_cleanup.parallel import TabletopCleanupTask
    raw = {'num_envs': 2, 'randomization': {'fixed_subset': ['speaker', 'lidar']}}
    raw.update(changes)
    task = TabletopCleanupTask(raw)
    backend = SimpleNamespace(ready=np.ones(2, bool), ticks=np.zeros(2, dtype=int), reset_specs=[None]*2)
    engine = initialize_cleanup_fixture(task, backend)
    specs = {i: task.sample_reset(i, 12, {}) for i in range(2)}
    backend.reset_specs[:] = [deepcopy(specs[i]) for i in range(2)]
    task.reset_task([0, 1], specs)
    task.bodies.pose_writes.clear(); task.bodies.velocity_writes.clear(); task.bodies.mask_writes.clear()
    return task


def test_indexed_pair_switch_restores_selected_native_masks_without_touching_peer_or_clock():
    task = bound_cleanup()
    task.contacts[0]['finger_forces_n'][1] = [.1, .2]
    task.checkers[0].grasped_and_lifted[1:] = True
    task.last_outcomes[0] = {'peer': 'historical result'}
    task.checkers[1].grasped_and_lifted[1:] = True
    peer_state = deepcopy(task.diagnostic_state([0]))
    peer_checker, peer_contacts = task.checkers[0], deepcopy(task.contacts[0])
    spec = task.sample_reset(1, 16, {'objects': {
        'gamepad': {'center_xy': [.35, .04], 'yaw_deg': 20.},
        'lidar': {'center_xy': [.3, -.15], 'yaw_deg': -30.}}})
    result = task.reset_task([1], {1: spec})
    assert task.diagnostic_state([0]) == peer_state
    assert task.checkers[0] is peer_checker and task.checkers[0].grasped_and_lifted[1:].all()
    for key in peer_contacts:
        np.testing.assert_array_equal(task.contacts[0][key], peer_contacts[key])
    assert task.last_outcomes[0] == {'peer': 'historical result'}
    assert task.last_outcomes[1] == {} and not task.checkers[1].grasped_and_lifted.any()
    assert all(not value.any() for value in task.contacts[1].values())
    assert task.active_masks[1].tolist() == [True, False, True]
    assert task.bodies.pose_writes == task.bodies.velocity_writes == [task.rows[1].tolist()]
    assert [row[0] for row in task.bodies.mask_writes] == [task.rows[1].tolist()]*2
    np.testing.assert_array_equal(task.bodies.mask_writes[0][1], np.zeros((3, 1)))
    np.testing.assert_array_equal(task.bodies.mask_writes[1][1], [[0], [1], [0]])
    for j, kind in enumerate(KINDS):
        row = task.rows[1, j]
        assert not task.bodies.velocities[row].any()
        if spec['active_mask'][j]:
            expected_xy = (np.array(task.scene['robot_base_position_world'][:2])+
                spec['objects'][kind]['center_xy']+task.engine.offsets[1, :2])
            np.testing.assert_allclose(task.bodies.positions[row, :2], expected_xy)
            bottom = np.asarray(task.config['asset_specs'][j]['support_vertices'])[:, 2].min()
            assert task.bodies.positions[row, 2]+bottom == pytest.approx(TABLE+.1+.0005)
        else:
            np.testing.assert_allclose(task.bodies.positions[row], task.parked_positions[1, j])
    assert task.engine.world.current_time == 4. and not task.engine.backend.ticks.any()
    assert result[1]['zero_tick_reset'] is True
    builtin_values(result)
    json.dumps(result, allow_nan=False)


def cleanup_contact(task, eid, first, second, *, force=.3, normal=(0., 0., 1.),
                    separation=0., direction=(0., 0., 1.), collider_only=False):
    paths = [first, second, first+'/Geometry/part', second+'/Geometry/part']
    if collider_only:
        paths[:2] = ['/World/Unknown0', '/World/Unknown1']
    impulse = np.asarray(direction)*force*task.execution['physics_dt']
    task.on_contact(eid, paths, [SimpleNamespace(impulse=impulse, normal=normal, separation=separation)])


@pytest.mark.parametrize('collider_only', [False, True])
def test_compound_root_or_child_contacts_classify_exact_active_objects_and_box_floor(collider_only):
    task = bound_cleanup()
    speaker, lidar = task.object_paths['speaker'], task.object_paths['lidar']
    cleanup_contact(task, 1, speaker, '/World/Piper/link7', force=.1, collider_only=collider_only)
    cleanup_contact(task, 1, speaker, '/World/Piper/link8', force=.2, collider_only=collider_only)
    cleanup_contact(task, 1, lidar, task.table_path, force=.3, collider_only=collider_only)
    cleanup_contact(task, 1, speaker, task.bin_floor_path, force=.4, collider_only=collider_only)
    cleanup_contact(task, 1, speaker, lidar, force=.5, collider_only=collider_only)
    cleanup_contact(task, 1, speaker+'_other', '/World/Piper/link7', force=99., collider_only=collider_only)
    cleanup_contact(task, 1, speaker, '/World/TabletopCleanup/Bin/Wall', force=99., collider_only=collider_only)
    measured = task.contacts[1]
    np.testing.assert_allclose(measured['finger_forces_n'], [[0., 0.], [.1, .2], [0., 0.]])
    np.testing.assert_allclose(measured['table_forces_n'], [0., 0., .3])
    np.testing.assert_allclose(measured['bin_floor_forces_n'], [0., .4, 0.])
    np.testing.assert_allclose(measured['object_support_forces_n'], [[0., 0., 0.], [0., 0., .5], [0., .5, 0.]])
    assert all(not value.any() for value in task.contacts[0].values())
    task.before_tick([1])
    assert all(not value.any() for value in task.contacts[1].values())


@pytest.mark.parametrize('normal,separation,direction,accepted', [
    ((0, 0, 1), 0., (0, 0, 1), True), ((0, 0, -1), 0., (0, 0, -1), True),
    ((1, 0, 0), 0., (0, 0, 1), False), ((0, 0, 1), .02, (0, 0, 1), False),
    ((0, 0, 1), -.02, (0, 0, 1), False), ((0, 0, 1), 0., (1, 0, 0), False),
])
def test_only_near_surface_vertical_impulse_can_support_cleanup_objects(normal, separation, direction, accepted):
    task = bound_cleanup()
    cleanup_contact(task, 0, task.object_paths['speaker'], task.object_paths['lidar'],
                    normal=normal, separation=separation, direction=direction)
    assert task.contacts[0]['object_support_forces_n'][1, 2] == pytest.approx(.3 if accepted else 0.)
    assert task.contacts[0]['object_support_forces_n'][2, 1] == pytest.approx(.3 if accepted else 0.)


def test_inactive_or_unreset_contacts_cannot_leak_into_selected_outcome():
    task = bound_cleanup()
    gamepad = task.object_paths['gamepad']
    task.engine.backend.ready[1] = False
    cleanup_contact(task, 1, gamepad, '/World/Piper/link7')
    assert all(not value.any() for value in task.contacts[1].values())
    with pytest.raises(RuntimeError, match='Inactive'):
        cleanup_contact(task, 0, gamepad, '/World/Piper/link7')
    assert all(not value.any() for value in task.contacts[0].values())


@pytest.mark.parametrize('corruption', ['enabled', 'visible', 'pose', 'velocity'])
def test_inactive_simulation_rendering_or_motion_drift_is_detected(corruption):
    task = bound_cleanup()
    row = task.rows[0, 0]
    if corruption == 'enabled':
        task.bodies.flags[row, 0] = 0
    elif corruption == 'visible':
        task.engine.visible[0, 0] = True
    elif corruption == 'pose':
        task.bodies.positions[row, 0] += .1
    else:
        task.bodies.velocities[row, 0] = .1
    with pytest.raises(RuntimeError):
        task.after_tick([0])


def test_cleanup_metadata_observation_mask_and_fixed_instruction_are_json_native():
    from pressb.runtime.contract import validate_metadata
    task = bound_cleanup()
    metadata = validate_metadata(task.metadata(), 'simulation')
    assert metadata['task_id'] == 'tabletop_cleanup'
    assert metadata['active_object_count'] == 2 and metadata['object_slots'] == list(KINDS)
    assert metadata['task_features_dim'] == 3
    assert metadata['action']['names'] == [f'joint{i}' for i in range(1, 7)]+['gripper_width']
    assert metadata['action']['units'] == ['rad']*6+['m']
    assert metadata['fixed_bin']['size_xyz_m'] == [.222, .148, .049]
    assert task.execution['terminal_boundary'] == 'physics_tick'
    for row in task.observation_fields([0, 1]).values():
        assert row['task_features'] == [0., 1., 1.]
        assert row['task'] == metadata['task_instruction']
        builtin_values(row)
    with pytest.raises(ValueError, match='instruction'):
        task.sample_reset(0, 1, {'task': 'Pick up the third object.'})
    builtin_values(metadata)
    json.dumps(metadata, allow_nan=False)


def test_explicit_measured_bin_size_can_be_refined_without_mutating_default():
    original = cleanup_config()
    refined = cleanup_config(bin={'size_xyz_m': [.224, .150, .05]})
    assert refined['bin']['size_xyz_m'] == [.224, .150, .05]
    assert original['bin']['size_xyz_m'] == [.222, .148, .049]
    assert cleanup_config()['bin'] == original['bin']
    assert original['max_episode_seconds'] == 120.


def test_feasible_narrow_domain_retries_random_yaw_that_does_not_fit():
    from pressb.tasks.tabletop_cleanup.configuration import sample_objects
    config = cleanup_config(randomization={'fixed_subset': ['gamepad', 'speaker'],
        'x_range_m': [.28, .417]})
    # This measured hull fits near 90 degrees; the first seed-0 yaw does not.
    explicit = {'gamepad': {'center_xy': [.35, .03], 'yaw_deg': 90.},
                'speaker': {'center_xy': [.35, -.16], 'yaw_deg': 0.}}
    assert len(sample_objects(config, 0, 2, explicit)['objects']) == 2
    assert len(sample_objects(config, 0, 0)['objects']) == 2


def feed_cleanup(task, frame, *, eid=0):
    ids = np.flatnonzero(task.active_masks[eid])
    rows = task.rows[eid, ids]
    task.bodies.positions[rows] = frame['positions'][ids]+task.engine.offsets[eid]
    task.bodies.rotations[rows] = frame['quaternions_wxyz'][ids]
    task.bodies.velocities[rows] = frame['velocities'][ids]
    task.engine.widths[eid] = frame['gripper_width_m']
    task.contacts[eid] = deepcopy(frame['contacts'])
    task.engine.backend.ticks[eid] = round(frame['seconds']/task.execution['physics_dt'])
    return task.after_tick([eid])[eid]


@pytest.mark.parametrize('terminate', [True, False])
def test_actual_task_requires_both_physical_histories_and_rewards_success_once(terminate):
    task = bound_cleanup(terminate_on_success=terminate, reward={'success': 2., 'failure': -.5})
    mask, tray, specs = [False, True, True], task.scene['bin'], task.config['asset_specs']
    for step, index in enumerate((1, 2)):
        frame = in_bin(seconds=step*.02, mask=mask, tray=tray, specs=specs)
        frame['positions'][index] = [.25, -.1, TABLE+.09]
        frame['gripper_width_m'] = .02
        frame['contacts']['bin_floor_forces_n'][index] = 0.
        frame['contacts']['finger_forces_n'][index] = [.1, .1]
        assert not feed_cleanup(task, frame, eid=1)['success']
    results = [feed_cleanup(task, in_bin(seconds=float(t), mask=mask, tray=tray, specs=specs), eid=1)
               for t in np.linspace(.1, .8, 8)]
    assert results[-1]['success'] and results[-1]['terminated'] is terminate
    assert sum(row['reward'] for row in results) == 2.
    assert all(row['truncated'] is False for row in results)
    assert results[-1]['info']['physical_outcome']['objects']['gamepad'] == {'active': False}
    builtin_values(results)
    json.dumps(results, allow_nan=False)


def test_actual_task_drop_failure_is_only_for_active_object_and_preserves_peer():
    task = bound_cleanup(reward={'failure': -.7})
    peer = deepcopy(task.diagnostic_state([0]))
    row = task.rows[1, 1]
    task.bodies.positions[row, 2] = TABLE+.1-.5
    task.engine.backend.ticks[1] = 1
    result = task.after_tick([1])[1]
    assert result['terminated'] and not result['truncated'] and not result['success']
    assert result['reason'] == 'object_dropped' and result['reward'] == -.7
    assert task.diagnostic_state([0]) == peer


class CleanupServo:
    def __init__(self, engine):
        self.engine = engine
        self.q = np.tile(engine.task.config['scene']['initial_state_joint_rad_m'], (2, 1)).astype(float)
        self.command = self.q.copy()

    def reset(self, ids, specs):
        for eid in ids:
            self.q[eid] = specs[eid].get('initial_state_joint_rad_m',
                self.engine.task.config['scene']['initial_state_joint_rad_m'])
            self.command[eid] = self.q[eid]

    def begin_action(self, ids, actions):
        self.command[ids] = actions
        return {eid: self.command[eid].tolist() for eid in ids}

    def before_tick(self, ids, substep, ticks_per_action):
        pass

    def observations(self, ids):
        return {eid: {'state': self.q[eid].tolist(), 'control_state': self.command[eid].tolist()} for eid in ids}

    def diagnostic_state(self, ids):
        return {eid: {'q': self.q[eid].tolist(), 'command': self.command[eid].tolist()} for eid in ids}


class CleanupCPUEngine:
    """Measured CPU dynamics only; all scheduling remains the production core."""
    def __init__(self, task, backend):
        self.task, self.backend = task, backend
        self.num_envs = task.execution['num_envs']
        self.closed, self.world_ticks = False, 0
        self.captures = []

    def initialize(self):
        measurement = initialize_cleanup_fixture(self.task, self.backend)
        for key, value in vars(measurement).items():
            if key != 'robot':
                setattr(self, key, value)
        self.task.engine = self
        self.robot = CleanupServo(self)

    def tick(self, live):
        self.world_ticks += 1
        self.world.current_time += self.task.execution['physics_dt']
        self.robot.q += (self.robot.command-self.robot.q)*.5
        # The native participation mask, not the scheduler's live list, controls
        # bodies. A held terminal slot can keep moving while inactive pool items
        # remain physically absent throughout all shared ticks.
        moving = np.flatnonzero(self.task.bodies.flags[:, 0] == 0)
        self.task.bodies.positions[moving, 0] += .00001

    def capture(self, ids, render_frames, reset_history=True):
        from pressb.runtime.contract import encode_image
        self.captures.append((list(ids), self.world_ticks))
        image = encode_image(np.full((2, 3, 3), self.world_ticks % 256, np.uint8))
        return {eid: {'up': image, 'wrist': image} for eid in ids}

    def publish_transforms(self):
        pass

    def reset_render_history(self):
        pass

    def check(self):
        if self.closed:
            raise RuntimeError('Closed fixture')

    def close(self):
        self.closed = True


def test_cleanup_real_shared_scheduler_preserves_peer_and_handles_timeout_without_isaac():
    from pressb.runtime.parallel import SharedWorldBackend
    from pressb.tasks.tabletop_cleanup.parallel import create_backend
    backend = create_backend({'num_envs': 2, 'horizon': 2, 'max_episode_seconds': .1,
        'randomization': {'fixed_subset': ['speaker', 'lidar']}}, engine_factory=CleanupCPUEngine)
    assert isinstance(backend, SharedWorldBackend)
    try:
        initial = backend.reset([0, 1], 3)
        assert len(initial) == 2 and backend.engine.world_ticks == 0
        before = deepcopy(backend.task.diagnostic_state([1]))
        body_before = backend.task.bodies.positions.copy()
        action = backend.engine.robot.q[0].tolist()
        action[0] += .01
        backend.step([{'env_id': eid, 'actions': [action]} for eid in (0, 1)])
        peer = deepcopy(backend.task.diagnostic_state([1]))
        robot_peer = deepcopy(backend.engine.robot.diagnostic_state([1]))
        assert peer != before and not np.array_equal(body_before, backend.task.bodies.positions)
        active = backend.task.active_masks.copy()
        for eid in range(2):
            for j in np.flatnonzero(~active[eid]):
                np.testing.assert_array_equal(backend.task.bodies.positions[backend.task.rows[eid, j]],
                                              backend.task.parked_positions[eid, j])
        tick = backend.engine.world_ticks
        backend.end_episode([0])
        backend.reset([0], 13, {'objects': {
            'gamepad': {'center_xy': [.35, .04], 'yaw_deg': 20.},
            'lidar': {'center_xy': [.3, -.15], 'yaw_deg': -30.}}})
        assert backend.engine.world_ticks == tick
        assert backend.task.diagnostic_state([1]) == peer
        assert backend.engine.robot.diagnostic_state([1]) == robot_peer
        rows = backend.step([{'env_id': eid, 'actions': [action, action]} for eid in (0, 1)])
        assert rows[1]['truncated'] and not rows[1]['terminated'] and not rows[1]['info']['success']
        assert rows[1]['info']['termination_reason'] == 'timeout'
        assert rows[1]['reward'] == 0.
        assert rows[1]['info']['physical_outcome']['active_mask'] == [False, True, True]
        for row in rows:
            builtin_values(row['info']['physical_outcome'])
        # Shared runtime timestamps may be np.float64 (supported by standard
        # JSON); the task-owned payload itself must use only builtin scalars.
        assert json.loads(json.dumps(rows, allow_nan=False)) == rows
    finally:
        backend.close()
    assert backend.engine.closed
