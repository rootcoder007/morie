"""Tests for morie.fn.mgwraic."""

import math

from morie.fn.mgwraic import mgwraic


def test_matches_rss_form():
    n, rss, tr = 40, 7.3, 6.2
    ll = -n / 2 * (math.log(2 * math.pi * rss / n) + 1)
    ref = n * math.log(rss / n) + n * math.log(2 * math.pi) + n * (n + tr) / (n - 2 - tr)
    assert abs(mgwraic(ll, tr, n).statistic - ref) < 1e-11
