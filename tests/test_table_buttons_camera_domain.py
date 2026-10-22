"""Full camera/table domain, checked against projections rather than its algebra."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import Delaunay

from pressb.table_buttons.camera_domain import camera_center_domain
from pressb.table_buttons.task import button_in_camera


def overhead_config():
    transform = np.eye(4)
    transform[2, 3] = 1.
    return dict(
        up_camera=dict(base_from_camera_usd=transform.tolist(), resolution=[101, 101],
                       fx=100., fy=100., cx=50., cy=50.),
        button_geometry=dict(base_radius_m=.02, base_height_m=.06, cap_height_m=.04),
        position_sampling=dict(pixel_margin=10., bounds_xy=[[0., .01], [0., .01]]),
        table_x_min=-1., table_depth=2., table_width=2.)


def test_overhead_domain_accounts_for_housing_radius_and_nearest_top_surface():
    result = camera_center_domain(overhead_config())
    # At z=.1 the half image width is .9 * 40/100 = .36 m.
    # Subtract the 2 cm housing radius, leaving centres at +/- .34 m.
    np.testing.assert_allclose(result["bounds_xy_m"], [[-.34, .34], [-.34, .34]], atol=1e-12)
    assert result["area_m2"] == pytest.approx(.68**2)
    assert result["geometry"]["top_z_m"] == pytest.approx(.1)


def test_finite_table_limits_full_housing_support_and_does_not_use_old_sampling_bounds():
    cfg = overhead_config()
    cfg.update(table_x_min=-.10, table_depth=.20, table_width=.30)
    before = deepcopy(cfg)
    result = camera_center_domain(cfg)
    assert cfg == before
    np.testing.assert_allclose(result["bounds_xy_m"], [[-.08, .08], [-.13, .13]], atol=1e-12)
    assert result["area_m2"] == pytest.approx(.16*.26)
    cfg["position_sampling"]["bounds_xy"] = [[20., 30.], [40., 50.]]
    assert camera_center_domain(cfg) == result


def test_oblique_real_camera_polygon_matches_full_prism_projection_across_extended_grid():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/piper_table_buttons.json").read_text())
    result = camera_center_domain(cfg)
    polygon = np.asarray(result["polygon_xy_m"])
    grid = np.array([[x, y] for x in np.arange(-.10, .91, .02)
                     for y in np.arange(-.78, .79, .02)])
    analytic = Delaunay(polygon).find_simplex(grid) >= 0
    radius = cfg["button_geometry"]["base_radius_m"]
    top = cfg["button_geometry"]["base_height_m"] + cfg["button_geometry"]["cap_height_m"]
    limits = np.asarray(result["table_center_limits_xy_m"])
    projected = [button_in_camera(point, cfg["up_camera"], button_radius=radius,
                                 button_top_z=top, margin_px=cfg["position_sampling"]["pixel_margin"])
                 and np.all(point >= limits[:, 0]) and np.all(point <= limits[:, 1])
                 for point in grid]
    np.testing.assert_array_equal(analytic, projected)
    assert result["area_m2"] > .20  # Expands beyond the old 9 x 28 cm rectangle.
    assert min(polygon[:, 0]) < .10 and max(polygon[:, 0]) > .50


@pytest.mark.parametrize("change", ["behind_camera", "no_table_support", "no_image_interior"])
def test_empty_intersection_returns_explicit_empty_domain(change):
    cfg = overhead_config()
    if change == "behind_camera":
        cfg["up_camera"]["base_from_camera_usd"][2][3] = -1.
    elif change == "no_table_support":
        cfg["table_width"] = .03
    else:
        cfg["position_sampling"]["pixel_margin"] = 51.
    result = camera_center_domain(cfg)
    assert result["polygon_xy_m"] == []
    assert result["bounds_xy_m"] is None and result["area_m2"] == 0.


def test_camera_looking_away_from_finite_table_returns_empty_domain():
    cfg = overhead_config()
    cfg["up_camera"]["base_from_camera_usd"][0][3] = 20.
    assert camera_center_domain(cfg)["polygon_xy_m"] == []


def test_invalid_camera_transform_raises_clear_error():
    cfg = overhead_config()
    cfg["up_camera"]["base_from_camera_usd"][2] = [0., 0., 0., 0.]
    with pytest.raises(ValueError, match="invertible"):
        camera_center_domain(cfg)
