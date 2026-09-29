"""Tests for morie.fn.binomial_variance: recompute Morin (2016) from the formula."""

import math

from morie.fn.binomial_variance import binomial_variance


def test_npq():
    for n, p in ((10, 0.3), (57, 0.5), (3, 0.0)):
        assert abs(binomial_variance(n, p)["variance"] - n * p * (1 - p)) < 1e-12
    # equals the pmf second central moment
    n, p = 12, 0.35
    pmf = [math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]
    mu = math.fsum(k * q for k, q in enumerate(pmf))
    var = math.fsum((k - mu) ** 2 * q for k, q in enumerate(pmf))
    assert abs(binomial_variance(n, p)["variance"] - var) < 1e-12
