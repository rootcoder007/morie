"""Tests for morie.fn.poisson_zero_series: recompute Morin (2016) from the formula."""

import math

from morie.fn.poisson_zero_series import poisson_zero_series


def test_partial_sums():
    a = 1.5
    r = poisson_zero_series(a, 20)
    s = 0.0
    for j in range(20):
        s += (-a) ** j / math.factorial(j)
        assert abs(r["partial_sums"][j] - s) < 1e-15
    assert abs(r["partial_sums"][-1] - math.exp(-a)) < 1e-12
    assert r["e_minus_a"] == math.exp(-a)
