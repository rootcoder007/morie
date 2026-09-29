"""Tests for morie.fn.minorm: the composite of E[mi], Var[mi] and the normal tail."""

import math

from morie.fn.minorm import minorm


def test_path_graph():
    n, s0, s1, s2, mi = 4, 6.0, 12.0, 40.0, 0.2
    e = -1 / (n - 1)
    v = (n * n * s1 - n * s2 + 3 * s0 * s0) / (s0 * s0 * (n * n - 1)) - 1 / (n - 1) ** 2
    z = (mi - e) / math.sqrt(v)
    r = minorm(mi, n, s0, s1, s2)
    assert abs(r.statistic - z) < 1e-14
    assert abs(r.p_value - 0.5 * math.erfc(z / math.sqrt(2))) < 1e-15
    assert abs(r.variance - v) < 1e-15
