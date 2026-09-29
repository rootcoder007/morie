"""Tests for morie.fn.poisson_binomial_peak_ratio: recompute Morin (2016) from the formula."""

import math

from morie.fn.poisson_binomial_peak_ratio import poisson_binomial_peak_ratio


def test_ratio_and_limit():
    n, p = 1000, 0.3
    k = round(p * n)
    pp = math.exp(k * math.log(p * n) - p * n - math.lgamma(k + 1))
    pb = math.exp(
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log(1 - p)
    )
    r = poisson_binomial_peak_ratio(n, p)
    assert abs(r["ratio"] - pp / pb) < 1e-10
    assert abs(r["sqrt_1_minus_p"] - math.sqrt(0.7)) < 1e-15
    assert abs(r["ratio"] - math.sqrt(0.7)) < 1e-3
