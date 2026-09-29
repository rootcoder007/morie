"""Tests for morie.fn.pwr_p: the Fleiss and Cohen-h power formulas recomputed."""

import math

from morie.fn.pwr_p import power_prop_test


def _phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _zq(p):
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_fleiss_formula():
    n, p1, p2 = 80, 0.4, 0.6
    z = _zq(0.975)
    pb = (p1 + p2) / 2
    s0, s1 = math.sqrt(2 * pb * (1 - pb)), math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    a = _phi((math.sqrt(n) * 0.2 - z * s0) / s1)
    b = 1 - _phi((math.sqrt(n) * 0.2 + z * s0) / s1)
    assert abs(power_prop_test(n=n, p1=p1, p2=p2, strict=False) - a) < 1e-12
    assert abs(power_prop_test(n=n, p1=p1, p2=p2) - (a + b)) < 1e-12


def test_cohen_h_and_solved_n():
    n, p1, p2 = 80, 0.4, 0.6
    h = abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))
    z = _zq(0.975)
    want = 1 - _phi(z - h * math.sqrt(n / 2)) + _phi(-z - h * math.sqrt(n / 2))
    assert abs(power_prop_test(n=n, p1=p1, p2=p2, method="cohen_h") - want) < 1e-12
    m = power_prop_test(p1=p1, p2=p2, power=0.85)
    assert abs(power_prop_test(n=m, p1=p1, p2=p2) - 0.85) < 1e-10
