"""Spacing profiles constrain every pair without changing legacy defaults."""
from copy import deepcopy

import numpy as np
import pytest

from pressb.tasks.table_buttons.spacing import resolve_spacing
from pressb.tasks.table_buttons.task import COLORS, button_in_camera, sample_button_positions


PROFILES = [("near", .06, .10), ("medium", .10, .18), ("far", .18, .30)]


def pairwise_distances(positions):
    points = np.asarray([positions[color] for color in COLORS])
    return np.asarray([np.linalg.norm(points[i] - points[j])
                       for i in range(3) for j in range(i + 1, 3)])


def camera():
    pose = np.eye(4)
    pose[2, 3] = 1.
    return dict(base_from_camera_usd=pose.tolist(), resolution=[640, 480],
                fx=500., fy=500., cx=320., cy=240.)


@pytest.mark.parametrize("profile,minimum,maximum", PROFILES)
def test_named_profile_constrains_both_ends_of_center_distance(profile, minimum, maximum):
    spacing = resolve_spacing(dict(spacing_profile=profile))
    assert spacing.profile == profile
    assert spacing.min_separation == minimum
    assert spacing.max_separation == maximum
    assert spacing.accepts(minimum) and spacing.accepts(maximum)
    assert not spacing.accepts(minimum - 1e-6)
    assert not spacing.accepts(maximum + 1e-6)


def test_no_profile_preserves_custom_minimum_and_unbounded_maximum():
    spacing = resolve_spacing(dict(min_separation=.05))
    assert spacing.profile is None
    assert spacing.min_separation == .05
    assert spacing.max_separation is None
    assert spacing.accepts(.05) and spacing.accepts(10.)
    assert not spacing.accepts(.049)


def test_custom_bounds_without_profile_are_supported():
    spacing = resolve_spacing(dict(min_separation=.07, max_separation=.12))
    assert spacing.accepts(.09) and not spacing.accepts(.13)


def test_selected_profile_overrides_legacy_bounds_without_mutating_configuration():
    config = dict(spacing_profile="near", min_separation=.20, max_separation=.30,
                  spacing_profiles={"near": dict(min_separation=.07, max_separation=.095)})
    original = deepcopy(config)
    spacing = resolve_spacing(config)
    assert (spacing.min_separation, spacing.max_separation) == (.07, .095)
    assert config == original


@pytest.mark.parametrize("config", [
    dict(spacing_profile="unknown"),
    dict(spacing_profile=True),
    dict(min_separation=.04),
    dict(min_separation=np.nan),
    dict(min_separation=np.inf),
    dict(max_separation=.09),
    dict(max_separation=np.nan),
    dict(max_separation=np.inf),
    dict(spacing_profile="near", spacing_profiles={"near": dict(min_separation=.08, max_separation=.07)}),
    dict(spacing_profile="near", spacing_profiles={"near": dict(min_separation=.04, max_separation=.10)}),
])
def test_invalid_spacing_does_not_silently_fall_back(config):
    with pytest.raises(ValueError):
        resolve_spacing(config)


def test_profile_minimum_must_keep_custom_button_housings_disjoint():
    with pytest.raises(ValueError, match="disjoint"):
        resolve_spacing(dict(spacing_profile="near"), button_radius=.035)


@pytest.mark.parametrize("profile,minimum,maximum", PROFILES)
def test_rectangle_sampler_applies_profile_to_all_pairs_and_preserves_visibility(profile, minimum, maximum):
    kwargs = dict(bounds_xy=((- .30, .30), (- .25, .25)),
                  spacing_profile=profile, max_attempts=10000, margin_px=16.)
    first = sample_button_positions(camera(), 12, **kwargs)
    same = sample_button_positions(camera(), 12, **kwargs)
    different = sample_button_positions(camera(), 13, **kwargs)
    assert first == same and first != different
    for seed in range(5):
        sampled = sample_button_positions(camera(), seed, **kwargs)
        distances = pairwise_distances(sampled)
        assert np.all(distances >= minimum) and np.all(distances <= maximum)
        assert all(button_in_camera(xy, camera(), margin_px=16.) for xy in sampled.values())


def test_rectangle_sampler_honors_profile_configuration_override():
    sampled = sample_button_positions(camera(), 10, bounds_xy=((- .3, .3), (- .25, .25)),
        spacing_profile="medium", spacing_profiles={"medium": dict(min_separation=.07, max_separation=.10)},
        max_attempts=10000)
    distances = pairwise_distances(sampled)
    assert np.all(distances >= .07) and np.all(distances <= .10)


def test_rectangle_sampler_impossible_profile_fails_within_proposal_budget():
    with pytest.raises(ValueError, match="Cannot place"):
        sample_button_positions(camera(), 0, bounds_xy=((0., .05), (0., .05)),
                                spacing_profile="far", max_attempts=20)
