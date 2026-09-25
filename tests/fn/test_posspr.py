"""Tests for posspr.posterior_predictive (DP mixture, Muller & Quintana 2004)."""

import math

import pytest

from morie.fn.posspr import posterior_predictive


def _k(y, th):
    m, s = th
    return math.exp(-0.5 * ((y - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def _g0(y):
    # prior predictive of a N(m, 1) kernel with m ~ N(0, 4): N(0, 5)
    return math.exp(-0.5 * y * y / 5.0) / math.sqrt(2 * math.pi * 5.0)


def test_posspr_basic():
    """p(y | data) = sum_j n_j / (alpha + n) k(y | theta_j)
    + alpha / (alpha + n) * int k dG0, recomputed on a grid."""
    grid = [-3.0, -0.5, 0.0, 1.2, 4.0]
    params = [(-1.0, 0.5), (2.0, 1.0)]
    counts = [7, 3]
    alpha = 1.5
    r = posterior_predictive(grid, params, counts, alpha, _k, _g0)
    assert isinstance(r, dict)
    n = sum(counts)
    exp = [sum(c / (alpha + n) * _k(y, th) for c, th in zip(counts, params))
           + alpha / (alpha + n) * _g0(y) for y in grid]
    assert r["density"] == pytest.approx(exp, rel=1e-14)
    assert r["new_cluster_weight"] == pytest.approx(alpha / (alpha + n), rel=1e-15)


def test_posspr_edge():
    """The predictive is a density: a Riemann sum over a wide fine grid
    integrates to 1 (tails beyond +-12 are < 1e-7, the midpoint rule
    error on this smooth integrand is < 1e-6)."""
    h = 0.01
    grid = [-12.0 + h * (i + 0.5) for i in range(2400)]
    r = posterior_predictive(grid, [(-1.0, 0.5), (2.0, 1.0)], [7, 3], 1.5, _k, _g0)
    assert sum(r["density"]) * h == pytest.approx(1.0, abs=1e-6)
