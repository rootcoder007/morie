"""Tests for geosample: geodesy and spatial sampling."""

import math

from morie.fn._rng import random_uniform
from morie.fn.geosample import (
    block_cv_folds,
    ecef_to_geodetic,
    geodetic_to_ecef,
    helmert_transform,
    kmeans_cv_folds,
    nested_grid,
    rhumb_line,
    temporal_stratified_sample,
)


def test_rhumb_line_equator_and_meridian():
    r = rhumb_line(0.0, 0.0, 0.0, -30.0)
    assert abs(r.distance - 6378137.0 * math.pi / 6) <= 1e-6 and abs(r.bearing - 270.0) <= 1e-12
    m = rhumb_line(10.0, 5.0, 40.0, 5.0)
    assert abs(m.distance - 6378137.0 * math.radians(30.0)) <= 1e-6 and m.bearing == 0.0
    # a rhumb line is never shorter than the great circle
    p1, p2 = math.radians(60), math.radians(62)
    gc = (
        2
        * 6378137.0
        * math.asin(
            math.sqrt(math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(15) / 2) ** 2)
        )
    )
    assert rhumb_line(60.0, -170.0, 62.0, 175.0).distance >= gc


def test_ecef_round_trip_and_known_point():
    x = geodetic_to_ecef(90.0, 0.0, 0.0)
    b = 6378137.0 * (1 - 1 / 298.257223563)
    assert abs(x[2] - b) <= 1e-6
    lat, lon, h = ecef_to_geodetic(*geodetic_to_ecef(-33.9, 151.2, 20.0, ellipsoid="Intl1924"), ellipsoid="Intl1924")
    assert abs(lat + 33.9) <= 1e-10 and abs(lon - 151.2) <= 1e-10 and abs(h - 20.0) <= 1e-6


def test_helmert_iogp_example_and_inverse_convention():
    out = helmert_transform([3657660.66, 255768.55, 5201382.11], 0, 0, 4.5, 0, 0, 0.554, 0.219)
    for a, b in zip(out, [3657660.78, 255778.43, 5201387.75]):
        assert abs(a - b) <= 0.015
    cf = helmert_transform([1.0e6, 2.0e6, 3.0e6], 1, 2, 3, 0.5, -0.3, 0.2, 1.5, convention="coordinate_frame")
    pv = helmert_transform([1.0e6, 2.0e6, 3.0e6], 1, 2, 3, -0.5, 0.3, -0.2, 1.5)
    assert max(abs(a - b) for a, b in zip(cf, pv)) <= 1e-9


def test_block_folds_follow_the_permutation():
    pts = [(i % 4, i // 4) for i in range(16)]
    r = block_cv_folds(pts, 2, 2, 2, seed=1)
    u = random_uniform(4, seed=1, stream=0)
    perm = [0, 1, 2, 3]
    for i in range(3, 0, -1):
        j = int(float(u[i]) * (i + 1))
        perm[i], perm[j] = perm[j], perm[i]
    for b, f in zip(r.block, r.fold):
        assert f == perm.index(b) % 2


def test_kmeans_folds_and_stratified_sample_and_grid():
    r = kmeans_cv_folds([(0, 0), (0.2, 0.1), (9, 9), (9.1, 9.3), (0.1, 0.3)], 2)
    assert r.fold[0] == r.fold[1] == r.fold[4] != r.fold[2] == r.fold[3]
    s = temporal_stratified_sample([i * 0.37 % 11 for i in range(50)], 13, n_strata=4, seed=3)
    assert sum(s.allocation) == 13 and len(set(s.index)) == 13
    assert abs(sum(s.weight) - 50) <= 1e-9
    g = nested_grid([(0.1, 0.1), (0.9, 0.9), (0.6, 0.2)], 2, bbox=(0, 1, 0, 1))
    assert all(sum(c) == 3 for c in g.counts)
    for fine, coarse in zip(g.cells[2], g.cells[1]):
        assert (fine % 4) // 2 + 2 * ((fine // 4) // 2) == coarse
