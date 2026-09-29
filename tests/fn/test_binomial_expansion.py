"""Tests for morie.fn.binomial_expansion: recompute Morin (2016) from the formula."""

import math

from morie.fn.binomial_expansion import binomial_expansion


def test_terms_and_sum():
    a, b, n = 1.5, -0.4, 6
    r = binomial_expansion(a, b, n)
    for k in range(n + 1):
        assert abs(r["terms"][k] - math.comb(n, k) * a ** (n - k) * b**k) < 1e-12
    assert abs(r["sum"] - (a + b) ** n) < 1e-12
    assert r["direct"] == (a + b) ** n
