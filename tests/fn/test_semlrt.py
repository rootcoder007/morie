"""Tests for morie.fn.semlrt: every expected value is recomputed from the formula."""

import math

from morie.fn.semlrt import semlrt


def _chisq_upper(x, df):
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    return math.exp(-x / 2)  # df = 2


def test_statistic_and_pvalue_default_df():
    r = semlrt(-10.2, -13.9)
    assert abs(r.statistic - 2 * (-10.2 + 13.9)) < 1e-14
    assert abs(r.p_value - _chisq_upper(r.statistic, 1)) < 1e-12
    assert r.extra["df"] == 1


def test_other_df():
    r = semlrt(-5.0, -7.25, df=2)
    assert abs(r.statistic - 4.5) < 1e-14
    assert abs(r.p_value - math.exp(-4.5 / 2)) < 1e-12
