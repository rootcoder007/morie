"""Tests for morie.fn.gaussian_approx_biased: recompute Morin (2016) from the formula."""

import math

from morie.fn.gaussian_approx_biased import gaussian_approx_biased


def test_formula():
    x, n, p = 4, 100, 0.3
    npq = n * p * (1 - p)
    ref = math.exp(-x * x / (2 * npq)) / math.sqrt(2 * math.pi * npq)
    assert abs(gaussian_approx_biased(x, n, p)["PG"] - ref) < 1e-15
    assert abs(ref - math.comb(n, 34) * p**34 * (1 - p) ** 66) < 2e-3
