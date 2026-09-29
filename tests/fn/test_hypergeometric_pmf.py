"""Tests for morie.fn.hypergeometric_pmf: recompute Morin (2016) from the formula."""

import math

from morie.fn.hypergeometric_pmf import hypergeometric_pmf


def test_pmf():
    N, K, n = 20, 7, 5
    for k in range(6):
        ref = math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)
        assert abs(hypergeometric_pmf(k, N, K, n)["probability"] - ref) < 1e-15
    assert abs(math.fsum(hypergeometric_pmf(k, N, K, n)["probability"] for k in range(6)) - 1) < 1e-14
