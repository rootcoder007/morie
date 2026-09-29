"""Tests for morie.fn.semspec: every expected value is recomputed from the formula."""

import math

from morie.fn.semspec import semspec


def test_common_factor_wald():
    V = [[0.01, 0.0, 0.0], [0.0, 0.04, 0.01], [0.0, 0.01, 0.09]]
    g = -0.2 + 0.4 * 1.0
    G = [1.0, 0.4, 1.0]
    s = sum(G[i] * V[i][j] * G[j] for i in range(3) for j in range(3))
    r = semspec([1.0], [-0.2], V, 0.4)
    assert abs(r.statistic - g * g / s) < 1e-13
    assert abs(r.p_value - math.erfc(math.sqrt(g * g / s / 2))) < 1e-12
