"""Tests for morie.fn.binomial_pmf_vector: recompute Morin (2016) from the formula."""

import math

from morie.fn.binomial_pmf_vector import binomial_pmf_vector


def test_pmf_values():
    n, p = 9, 0.37
    pmf = binomial_pmf_vector(n, p)["pmf"]
    for k in range(n + 1):
        assert abs(pmf[k] - math.comb(n, k) * p**k * (1 - p) ** (n - k)) < 1e-15
    assert abs(math.fsum(pmf) - 1.0) < 1e-14
