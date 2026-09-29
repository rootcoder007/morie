"""Tests for morie.fn.poisson_small_interval: recompute Morin (2016) from the formula."""

import math

from morie.fn.poisson_small_interval import poisson_small_interval


def test_first_order():
    lam, eps = 3.0, 1e-3
    r = poisson_small_interval(lam, eps)
    assert r["approx"] == lam * eps
    assert abs(r["exact"] - lam * eps * math.exp(-lam * eps)) < 1e-16
    assert abs(r["abs_error"] - abs(r["approx"] - r["exact"])) < 1e-18
