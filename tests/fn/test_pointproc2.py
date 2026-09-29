"""Tests for pointproc2: point-process simulation, intensity fits and space-time summaries."""

import math

from morie.fn._rng import random_uniform
from morie.fn.pointproc2 import (
    abramson_intensity,
    area_interaction_simulate,
    berman_turner_fit,
    lgcp_simulate_grid,
    st_g_function,
    st_j_function,
    st_k_function,
)


def test_area_interaction_with_eta_one_is_poisson():
    ns = [area_interaction_simulate(30.0, 1.0, 0.05, (0, 1, 0, 1), 400, grid=10, seed=s).n for s in range(20)]
    assert abs(sum(ns) / 20 - 30.0) < 4 * math.sqrt(30.0 / 20) + 3


def test_lgcp_counts_follow_intensity():
    r = lgcp_simulate_grid(3, 2, (0, 3, 0, 2), 1.0, 0.5, 1.0, seed=3)
    assert all(c >= 0 for c in r.counts) and all(
        abs(math.log(v) - 1.0 - z) <= 1e-12 for v, z in zip(r.intensity, r.field)
    )


def test_abramson_bandwidths_follow_square_root_law():
    pts = [(0.5 + 0.3 * math.sin(i * 1.3), 0.5 + 0.3 * math.cos(i * 0.7)) for i in range(25)]
    r = abramson_intensity(pts, pts[:2], 0.1)
    g = math.exp(sum(math.log(v) for v in r.pilot) / 25)
    for p, h in zip(r.pilot, r.bandwidths):
        assert abs(h - 0.1 * min((p / g) ** -0.5, 5.0)) <= 1e-15


def test_berman_turner_intercept_only_is_count_over_area():
    pts = [(0.1, 0.2), (0.4, 0.8), (0.7, 0.5), (0.9, 0.9), (0.3, 0.3)]
    r = berman_turner_fit(pts, (0, 1, 0, 2), lambda x, y: [], 3, 3)
    assert abs(math.exp(r.coefficients[0]) - 5 / 2) <= 1e-12


def test_space_time_k_g_j_by_hand():
    P = [(0, 0, 0), (1, 0, 1), (0, 1, 5), (3, 3, 3)]
    k = st_k_function(P, [1.0], [1.0], 16.0, 5.0)
    assert abs(k.K[0][0] - 16 * 5 * 2 / 12) <= 1e-12
    assert abs(st_k_function(P, [1.0], [1.0], 16.0, 5.0, method="stpp").K[0][0] - 16 * 5 * 2 / 16) <= 1e-12
    assert st_g_function(P, [1.0], [1.0]) == [[0.5]]
    j = st_j_function(P, [1.0], [1.0], (0, 4, 0, 4), (0, 5), n_grid=4, n_time=5)
    assert abs(j.J[0][0] - 0.5 / (1 - j.F[0][0])) <= 1e-15
    assert random_uniform(1, seed=0, stream=0) is not None
