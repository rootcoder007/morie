"""Tests for morie.fn.poisson_mean_rate: recompute Morin (2016) from the formula."""

import math

from morie.fn.poisson_mean_rate import poisson_mean_rate


def test_series_mean():
    lam, t = 2.5, 4.0
    a = lam * t
    series = math.fsum(k * math.exp(-a) * a**k / math.factorial(k) for k in range(80))
    assert abs(poisson_mean_rate(lam, t)["expected_events"] - a) < 1e-15
    assert abs(series - a) < 1e-12
