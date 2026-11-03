"""Target-oriented routes must not gain diversity by sideways/backward detours."""
from copy import deepcopy

import numpy as np
import pytest

from pressb.table_buttons.motion import plan_press, screening_path
from pressb.table_buttons.motion_templates import ColorMotionTemplates
from pressb.table_buttons.target_relative_motion import targetward_metrics


class CartesianKinematics:
    lower = np.full(6, -2.)
    upper = np.full(6, 2.)
    velocity_limits = np.full(6, 3.)

    def fk(self, q):
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        return pose

    def solve(self, position, seed, rotation):
        return np.r_[position, np.zeros(3)]


def config(**options):
    return dict(initial_joint_positions_rad=[.20, -.02, .16, 0., 0., 0.], fps=30,
                motion_templates=dict(mode='target_relative_v2', **options))


def approach(xy=(.35, .12)):
    hover, near, press = [np.r_[xy, z] for z in (.12, .043, .019)]
    return dict(orientation=np.eye(3), hover=hover, near=near, press=press, pitch_deg=160.,
                q_hover=np.r_[hover, np.zeros(3)],
                q_path=np.array([np.r_[xyz, np.zeros(3)] for xyz in (hover, near, press)]))


@pytest.mark.parametrize('xy', [(.35, .12), (.15, .2), (.16, -.15), (.18, -.015)])
def test_routes_move_along_target_direction_for_different_target_locations(xy):
    kin, cfg = CartesianKinematics(), config()
    templates = ColorMotionTemplates(kin, cfg)
    initial = np.array(cfg['initial_joint_positions_rad'])
    for color in ('red', 'yellow', 'blue'):
        selected = templates.attach(approach(xy), color)
        templates.validate(selected, color)
        points = np.array(selected['transit_path']['tip_xyz_m'])
        actual = np.array(selected['transit_path']['joints_rad'])[:, :3]
        np.testing.assert_allclose(actual, points, atol=1e-12)
        np.testing.assert_allclose(points[0], initial[:3], atol=1e-12)
        np.testing.assert_allclose(points[-1], selected['near'], atol=1e-12)
        direction = targetward_metrics(actual, points[0], points[-1])
        assert max(direction.values()) < 1e-12
        assert max(selected['motion_template']['targetward_check'].values()) < 1e-12
        # The discrete collision path must contain every commanded prefix point.
        checked = screening_path(initial, selected)
        for q in selected['transit_path']['joints_rad']:
            assert np.linalg.norm(checked-q, axis=1).min() < 1e-12


def test_three_styles_have_distinct_height_profiles_without_sideways_waypoints():
    kin, cfg = CartesianKinematics(), config()
    templates = ColorMotionTemplates(kin, cfg)
    routes = {c: templates.attach(approach(), c) for c in ('red', 'yellow', 'blue')}
    red, yellow, blue = [np.array(routes[c]['transit_path']['tip_xyz_m']) for c in routes]
    delta = red[-1]-red[0]
    assert np.max(np.linalg.norm(np.cross(red-red[0], delta), axis=1)) < 1e-12
    assert yellow[54, 2] == pytest.approx(red[0, 2]+.06)
    np.testing.assert_allclose(yellow[54, :2], red[0, :2]+.4*(red[-1, :2]-red[0, :2]))
    assert np.all(np.diff(yellow[:55, 2]) >= -1e-12)
    assert np.all(np.diff(yellow[54:, 2]) <= 1e-12)
    np.testing.assert_allclose(blue[72:, :2], np.tile(blue[-1, :2], (37, 1)))
    assert blue[72, 2] == pytest.approx(.18)
    assert np.all(np.diff(blue[72:, 2]) <= 1e-12)
    assert templates.diversity(cfg['initial_joint_positions_rad'], routes)['accepted']


def test_press_starts_after_cartesian_prefix_and_retains_contact_and_return():
    kin, cfg = CartesianKinematics(), config()
    initial = np.r_[cfg['initial_joint_positions_rad'], .0002]
    templates = ColorMotionTemplates(kin, cfg)
    for color in ('red', 'yellow', 'blue'):
        selected = templates.attach(approach(), color)
        times, commands, report = plan_press(kin, initial, [.35, .12, .025], approach=selected)
        assert report['phase_times']['press_start'] == pytest.approx(4.)
        assert report['phase_times']['release_start'] == pytest.approx(5.35)
        assert report['phase_times']['return_hover_end'] == pytest.approx(6.45)
        assert times[-1] == pytest.approx(8.7)
        np.testing.assert_allclose(commands[120, :3], selected['near'])
        np.testing.assert_allclose(commands[150, :3], selected['press'])
        np.testing.assert_allclose(commands[-1], initial)
        np.testing.assert_array_equal(commands[:, 6], np.full(len(commands), initial[6]))


@pytest.mark.parametrize('mutation', ['color', 'target', 'joints', 'time', 'signature'])
def test_reject_stale_target_or_color_and_modified_cached_samples(mutation):
    templates = ColorMotionTemplates(CartesianKinematics(), config())
    selected = templates.attach(approach(), 'red')
    if mutation == 'color':
        selected['motion_template']['color'] = 'blue'
    elif mutation == 'target':
        selected['near'] = np.array([.4, .12, .043])
    elif mutation == 'joints':
        selected['transit_path']['joints_rad'][50][0] += .001
    elif mutation == 'time':
        selected['transit_path']['time_s'][50] += .01
    else:
        selected['motion_template']['config_signature'] = 'stale'
    with pytest.raises(ValueError, match='Stale|mismatched'):
        templates.validate(selected, 'red')


def test_route_cache_does_not_reuse_another_targets_path_or_expose_mutable_state():
    templates = ColorMotionTemplates(CartesianKinematics(), config())
    first = templates.attach(approach(), 'red')
    expected = deepcopy(first['transit_path'])
    first['transit_path']['joints_rad'][30][0] = 1.
    assert templates.attach(approach(), 'red')['transit_path'] == expected
    other = templates.attach(approach((.15, -.12)), 'red')
    assert other['transit_path'] != expected
    np.testing.assert_allclose(other['transit_path']['tip_xyz_m'][-1], [.15, -.12, .043])


@pytest.mark.parametrize('options', [dict(transit_duration_s=2.), dict(transit_duration_s=3.61),
    dict(mid_lift_m=-.1), dict(overhead_z_m=.05), dict(min_transit_separation_m=-.1),
    dict(mid_lift_m=float('nan')), dict(waypoints_base_m={'red': [0, 0, 0]})])
def test_invalid_relative_settings_are_not_silently_ignored(options):
    with pytest.raises(ValueError):
        ColorMotionTemplates(CartesianKinematics(), config(**options))


@pytest.mark.parametrize('mutation', ['initial', 'times', 'shape', 'endpoint', 'missing'])
def test_planner_rejects_malformed_prefix(mutation):
    kin, cfg = CartesianKinematics(), config()
    selected = ColorMotionTemplates(kin, cfg).attach(approach(), 'blue')
    path = selected['transit_path']
    if mutation == 'initial':
        path['joints_rad'][0][0] += .01
    elif mutation == 'times':
        path['time_s'][1] = 0.
    elif mutation == 'shape':
        path['joints_rad'][0] = [0.]*5
    elif mutation == 'endpoint':
        path['joints_rad'][-1][0] += .01
    else:
        del selected['transit_path']
    with pytest.raises(ValueError):
        plan_press(kin, np.r_[cfg['initial_joint_positions_rad'], .0002],
                   [.35, .12, .025], approach=selected)
