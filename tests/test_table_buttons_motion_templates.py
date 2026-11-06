"""Color detours separate motion without changing the physical press endpoint."""
from copy import deepcopy

import numpy as np
import pytest

from pressb.tasks.table_buttons.motion import plan_press, screening_path
from pressb.tasks.table_buttons.motion_templates import ColorMotionTemplates


COLORS = ('red', 'yellow', 'blue')


class CartesianTestKinematics:
    """Identity Cartesian joints isolate timing and path wiring from IK errors."""
    lower = np.full(6, -2.)
    upper = np.full(6, 2.)
    velocity_limits = np.full(6, 2.)

    def solve(self, xyz, seed, orientation):
        return np.r_[xyz, np.zeros(3)]

    def fk(self, q):
        result = np.eye(4)
        result[:3, 3] = np.asarray(q)[:3]
        return result


@pytest.fixture
def kin():
    return CartesianTestKinematics()


def config(**options):
    return dict(initial_joint_positions_rad=[.20, 0., .20, 0., 0., 0.], fps=30,
                motion_templates=dict(mode='color_detours_v1', **options))


def approach():
    hover = np.array([.35, 0., .12])
    near = np.array([.35, 0., .043])
    press = np.array([.35, 0., .019])
    return dict(orientation=np.eye(3), hover=hover, near=near, press=press,
                q_hover=np.r_[hover, np.zeros(3)], pitch_deg=160.,
                q_path=np.array([np.r_[xyz, np.zeros(3)] for xyz in (hover, near, press)]))


def test_legacy_shared_schedule_has_known_clock_knots_and_targets(kin):
    cfg = config()
    del cfg['motion_templates']
    templates = ColorMotionTemplates(kin, cfg)
    initial = np.r_[cfg['initial_joint_positions_rad'], .0002]
    selected = templates.attach(approach(), 'red')
    times, commands, report = plan_press(kin, initial, [.35, 0., .025], fps=60, approach=selected)
    assert not templates.enabled
    assert 'transit_waypoints_rad' not in selected
    assert len(times) == 469
    np.testing.assert_allclose(times, np.arange(469)/60, atol=0, rtol=0)
    # Frozen legacy schedule landmarks include each phase and midpoint, not just endpoints.
    expected = {
        0.: [.20, 0., .20], .4: [.20, 0., .20],
        1.3: [.275, 0., .16], 2.2: [.35, 0., .12],
        2.65: [.35, 0., .0815], 3.1: [.35, 0., .043],
        3.6: [.35, 0., .031], 4.1: [.35, 0., .019],
        4.45: [.35, 0., .019], 5.: [.35, 0., .0695],
        5.55: [.35, 0., .12], 6.45: [.275, 0., .16],
        7.35: [.20, 0., .20], 7.8: [.20, 0., .20],
    }
    for seconds, xyz in expected.items():
        np.testing.assert_allclose(commands[round(seconds*60)], np.r_[xyz, np.zeros(3), .0002], atol=1e-12)
    assert report['phase_times']['press_start'] == 3.1
    assert report['phase_times']['release_start'] == 4.45
    assert report['phase_times']['return_hover_end'] == 5.55


def test_three_colors_take_distinct_anchors_then_share_identical_press(kin):
    cfg = config()
    templates = ColorMotionTemplates(kin, cfg)
    initial = np.r_[cfg['initial_joint_positions_rad'], .0002]
    shared_times, shared_commands, _ = plan_press(kin, initial, [.35, 0., .025], approach=approach())
    approaches = {}
    for color in COLORS:
        selected = templates.attach(approach(), color)
        approaches[color] = selected
        templates.validate(selected, color)
        times, commands, report = plan_press(kin, initial, [.35, 0., .025], approach=selected)
        assert report['motion_template']['color'] == color
        assert times[-1] == pytest.approx(shared_times[-1]+3.)
        assert len(times) == 325
        assert report['phase_times']['press_start'] == pytest.approx(6.1)
        assert report['phase_times']['release_start'] == pytest.approx(7.45)
        assert report['phase_times']['return_hover_end'] == pytest.approx(8.55)
        # Full final-approach, contact and return suffix matches the old trajectory.
        np.testing.assert_allclose(commands[156:], shared_commands[66:], atol=1e-12)
        np.testing.assert_array_equal(commands[:, 6], np.full(len(times), .0002))
        np.testing.assert_allclose(kin.fk(commands[84])[:3, 3],
                                   templates.options['waypoints_base_m'][color], atol=1e-12)
        # Collision screening must visit the same detour and include the return home.
        screened = screening_path(initial, selected)
        assert np.min(np.linalg.norm(screened-selected['transit_waypoints_rad'][0], axis=1)) < 1e-12
        np.testing.assert_array_equal(screened[-1], initial[:6])
    diversity = templates.diversity(initial[:6], approaches)
    assert diversity['accepted']
    assert min(diversity['pairwise_mean_tip_distance_m'].values()) >= .10


def test_identical_color_routes_do_not_satisfy_separation_requirement(kin):
    cfg = config(waypoints_base_m={color: [.30, 0., .30] for color in COLORS})
    templates = ColorMotionTemplates(kin, cfg)
    selected = {color: templates.attach(approach(), color) for color in COLORS}
    result = templates.diversity(cfg['initial_joint_positions_rad'], selected)
    assert not result['accepted']
    assert all(value == 0. for value in result['pairwise_mean_tip_distance_m'].values())


@pytest.mark.parametrize('mutation', ['color', 'duration', 'signature', 'waypoint'])
def test_reusing_mismatched_cached_routes_is_rejected(kin, mutation):
    templates = ColorMotionTemplates(kin, config())
    selected = templates.attach(approach(), 'red')
    selected = deepcopy(selected)
    if mutation == 'color':
        selected['motion_template']['color'] = 'blue'
    elif mutation == 'duration':
        selected['motion_template']['leg_duration_s'] = 3.
    elif mutation == 'signature':
        selected['motion_template']['config_signature'] = 'stale'
    else:
        selected['transit_waypoints_rad'][0][0] += .001
    with pytest.raises(ValueError, match='Stale|mismatched'):
        templates.validate(selected, 'red')


def test_shared_config_rejects_a_cached_detour_instead_of_silently_replaying_it(kin):
    selected = ColorMotionTemplates(kin, config()).attach(approach(), 'red')
    cfg = config()
    cfg['motion_templates'] = {'mode': 'shared'}
    with pytest.raises(ValueError, match='shared'):
        ColorMotionTemplates(kin, cfg).attach(selected, 'red')


@pytest.mark.parametrize('options', [
    {'mode': 'unknown'}, {'mode': 'shared', 'leg_duration_s': 2.4},
    {'mode': 'color_detours_v1', 'unknown': True},
    {'mode': 'color_detours_v1', 'leg_duration_s': 1.7},
    {'mode': 'color_detours_v1', 'leg_duration_s': 2.405},
    {'mode': 'color_detours_v1', 'leg_duration_s': np.nan},
    {'mode': 'color_detours_v1', 'pitch_deg': 90},
    {'mode': 'color_detours_v1', 'min_transit_separation_m': 0},
    {'mode': 'color_detours_v1', 'min_transit_separation_m': np.inf},
    {'mode': 'color_detours_v1', 'waypoints_base_m': {'red': [.3, 0, .3]}},
    *[{'mode': 'color_detours_v1', 'waypoints_base_m': {
        'red': xyz, 'yellow': [.3, 0, .3], 'blue': [.3, .2, .3]}}
        for xyz in ([.3, 0], [[.3, 0, .3]], [.3, 0, .12], [.3, np.nan, .3])],
])
def test_bad_config_cannot_silently_enable_shared_or_unsafe_routes(kin, options):
    cfg = config()
    cfg['motion_templates'] = options
    with pytest.raises(ValueError):
        ColorMotionTemplates(kin, cfg)


@pytest.mark.parametrize('mutation', ['missing', 'shape', 'nan', 'limits', 'duration'])
def test_motion_planner_rejects_invalid_declared_detour(kin, mutation):
    cfg = config()
    selected = ColorMotionTemplates(kin, cfg).attach(approach(), 'red')
    if mutation == 'missing':
        selected['transit_waypoints_rad'] = []
    elif mutation == 'shape':
        selected['transit_waypoints_rad'] = [[.1]*5]
    elif mutation == 'nan':
        selected['transit_waypoints_rad'][0][0] = np.nan
    elif mutation == 'limits':
        selected['transit_waypoints_rad'][0][0] = 3.
    else:
        selected['motion_template']['leg_duration_s'] = 0.
    with pytest.raises(ValueError):
        plan_press(kin, np.r_[cfg['initial_joint_positions_rad'], .0002],
                   [.35, 0., .025], approach=selected)
