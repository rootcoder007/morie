"""Tests for morie.fn.sdmlrt: every expected value is recomputed from the formula."""

import math

from morie.fn.sdmlrt import sdmlrt


def _chisq_upper(x, df):
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    return math.exp(-x / 2)  # df = 2


def test_statistic_and_pvalue_default_df():
    r = sdmlrt(-10.2, -13.9)
    assert abs(r.statistic - 2 * (-10.2 + 13.9)) < 1e-14
    assert abs(r.p_value - _chisq_upper(r.statistic, 2)) < 1e-12
    assert r.extra["df"] == 2


def test_other_df():
    r = sdmlrt(-5.0, -7.25, df=2)
    assert abs(r.statistic - 4.5) < 1e-14
    assert abs(r.p_value - math.exp(-4.5 / 2)) < 1e-12
