"""Tests for morie.fn.sdmcf: every expected value is recomputed from the formula."""

import math

from morie.fn.sdmcf import sdmcf


def test_one_covariate_delta_method():
    V = [[0.01, 0.002, 0.0], [0.002, 0.04, 0.01], [0.0, 0.01, 0.09]]
    b, t, rho = 1.0, -0.2, 0.4
    g = t + rho * b
    G = [b, rho, 1.0]
    s = sum(G[i] * V[i][j] * G[j] for i in range(3) for j in range(3))
    r = sdmcf([b], [t], V, rho)
    assert abs(r.statistic - g * g / s) < 1e-13
    assert abs(r.p_value - math.erfc(math.sqrt(g * g / s / 2))) < 1e-12
    assert r.extra["df"] == 1


def test_restriction_holds_exactly():
    V = [[0.01, 0, 0, 0, 0], [0, 0.04, 0, 0, 0], [0, 0, 0.05, 0, 0], [0, 0, 0, 0.09, 0], [0, 0, 0, 0, 0.07]]
    r = sdmcf([1.0, -0.5], [-0.4, 0.2], V, 0.4)
    assert abs(r.statistic) < 1e-14 and abs(r.p_value - 1.0) < 1e-12
