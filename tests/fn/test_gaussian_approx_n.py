"""Tests for morie.fn.gaussian_approx_n: recompute Morin (2016) from the formula."""

import math

from morie.fn.gaussian_approx_n import gaussian_approx_n


def test_formula():
    for x, n in ((0, 1), (3, 100), (-2.5, 40)):
        assert abs(gaussian_approx_n(x, n)["PG"] - math.exp(-2 * x * x / n) / math.sqrt(math.pi * n / 2)) < 1e-15
    n, x = 400, 6
    assert abs(gaussian_approx_n(x, n)["PG"] - math.comb(n, n // 2 + x) / 2**n) < 1e-4
