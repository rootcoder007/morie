"""Tests for morie.fn.gaussian_approx_2n: recompute Morin (2016) from the formula."""

import math

from morie.fn.gaussian_approx_2n import gaussian_approx_2n


def test_formula():
    for x, n in ((0, 1), (3, 50), (-2, 20)):
        assert abs(gaussian_approx_2n(x, n)["PG"] - math.exp(-x * x / n) / math.sqrt(math.pi * n)) < 1e-15
    # it approximates C(2n, n + x) / 2^(2n)
    n, x = 200, 5
    assert abs(gaussian_approx_2n(x, n)["PG"] - math.comb(2 * n, n + x) / 4**n) < 1e-4
