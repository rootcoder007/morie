"""Tests for morie.fn.semwald: every expected value is recomputed from the formula."""

import math

from morie.fn.semwald import semwald


def test_wald_is_squared_z_with_normal_pvalue():
    est, se = 0.37, 0.12
    r = semwald(est, se)
    z = est / se
    assert abs(r.statistic - z * z) < 1e-12
    assert abs(r.extra["z"] - z) < 1e-14
    assert abs(r.p_value - math.erfc(abs(z) / math.sqrt(2))) < 1e-12


def test_negative_estimate():
    r = semwald(-0.2, 0.25)
    assert abs(r.statistic - 0.64) < 1e-14
