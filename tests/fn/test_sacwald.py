"""Tests for morie.fn.sacwald: every expected value is recomputed from the formula."""

import math

from morie.fn.sacwald import sacwald


def test_joint_wald_two_parameters():
    b = [0.3, -0.2]
    V = [[0.01, 0.002], [0.002, 0.02]]
    d = V[0][0] * V[1][1] - V[0][1] * V[1][0]
    Vi = [[V[1][1] / d, -V[0][1] / d], [-V[1][0] / d, V[0][0] / d]]
    w = sum(b[i] * Vi[i][j] * b[j] for i in range(2) for j in range(2))
    r = sacwald(b, V)
    assert abs(r.statistic - w) < 1e-12
    assert abs(r.p_value - math.exp(-w / 2)) < 1e-12
    assert r.extra["df"] == 2


def test_no_ridge_on_covariance():
    # a tiny variance must not be regularised away
    r = sacwald([1e-4], [[1e-10]])
    assert abs(r.statistic - 1e-8 / 1e-10) < 1e-6
