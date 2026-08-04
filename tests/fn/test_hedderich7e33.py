"""Tests for hedderich7e33 -- the Anderson-Darling A^2 statistic.

Anchored on the defining integral.  A^2 = n * Int_0^1 (F_n(u)-u)^2/(u(1-u)) du,
and on (u_k, u_{k+1}) the empirical cdf is the constant c = k/n, so
partial fractions give the exact antiderivative

    Int (c-u)^2 / (u(1-u)) du = -u + c^2 log(u) - (1-c)^2 log(1-u)

(the c^2 term vanishes at c = 0, the (1-c)^2 term at c = 1, so both
endpoints are finite).  Evaluating that sum is a completely separate
route to A^2 from the order-statistic formula the module implements.
"""

import math

import pytest

from morie.fn.hedderich7e33 import ad_statistic, hedderich_chapter_7_equation_33

GRID = [(i + 0.5) / 24.0 for i in range(24)]


def _ad_by_integral(u):
    us = sorted(float(v) for v in u)
    n = len(us)
    pts = [0.0] + us + [1.0]

    def F(x, c):
        v = -x
        if c != 0.0:
            v += c * c * math.log(x)
        if c != 1.0:
            v -= (1.0 - c) ** 2 * math.log(1.0 - x)
        return v

    return n * sum(
        F(pts[k + 1], k / n) - F(pts[k], k / n) for k in range(n + 1) if pts[k + 1] > pts[k]
    )


def test_matches_its_defining_integral():
    assert ad_statistic(GRID) == pytest.approx(_ad_by_integral(GRID), abs=1e-9)


def test_uniform_grid_is_small_bunched_sample_is_large():
    assert ad_statistic(GRID) < 0.5
    assert ad_statistic([0.9 + 0.005 * i for i in range(20)]) > 10.0


def test_wrapper_applies_the_cdf():
    res = hedderich_chapter_7_equation_33(GRID, lambda v: v)
    assert res["statistic"] == pytest.approx(ad_statistic(GRID))
    assert res["n"] == 24


def test_probability_zero_observation_is_infinite_not_clipped():
    assert ad_statistic([0.0, 0.3, 0.7]) == float("inf")
    assert ad_statistic([0.3, 0.7, 1.0]) == float("inf")


def test_needs_two_observations():
    with pytest.raises(ValueError):
        ad_statistic([0.5])
