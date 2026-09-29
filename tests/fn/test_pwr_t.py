"""Tests for morie.fn.pwr_t: power recomputed by numerical integration of the noncentral t."""

import math

from morie.fn.pwr_t import power_t_test


def _nct_sf(x, df, ncp, m=4000):
    # P(T' > x) = E_V[ Phi(ncp - x sqrt(V/df)) ], V ~ chi2(df), by Gauss-free midpoint quadrature on the chi density
    k = df / 2.0
    lg = math.lgamma(k)
    hi = df + 40 * math.sqrt(2 * df)
    h = hi / m
    s = 0.0
    for i in range(m):
        v = (i + 0.5) * h
        dens = math.exp((k - 1) * math.log(v) - v / 2 - k * math.log(2) - lg)
        s += dens * 0.5 * math.erfc(-(ncp - x * math.sqrt(v / df)) / math.sqrt(2))
    return s * h


def _t_quantile(p, df):
    # invert the central t cdf by bisection on the quadrature sf
    lo, hi = 0.0, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _nct_sf(mid, df, 0.0) > 1 - p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_two_sample_power_strict_and_not():
    n, d = 12, 1.1
    nu = 2 * (n - 1)
    qu = _t_quantile(0.975, nu)
    ncp = math.sqrt(n / 2) * d
    upper = _nct_sf(qu, nu, ncp)
    lower = 1 - _nct_sf(-qu, nu, ncp)
    assert abs(power_t_test(n=n, delta=d, strict=False) - upper) < 2e-6
    assert abs(power_t_test(n=n, delta=d) - (upper + lower)) < 2e-6


def test_solved_n_and_delta_reproduce_the_power():
    n = power_t_test(delta=0.8, power=0.9, type="one-sample")
    assert abs(power_t_test(n=n, delta=0.8, type="one-sample") - 0.9) < 1e-10
    d = power_t_test(n=15, power=0.8, sd=2.0, alternative="one-sided")
    assert abs(power_t_test(n=15, delta=d, sd=2.0, alternative="one-sided") - 0.8) < 1e-10


def test_exactly_one_missing():
    import pytest

    with pytest.raises(ValueError):
        power_t_test(n=10)
