"""Tests for morie.fn.sdmr2: every expected value is recomputed from the formula."""

import math

from morie.fn.sdmr2 import sdmr2


def test_nagelkerke_formula():
    l1, l0, n = -20.0, -31.5, 40
    cs = 1 - math.exp(2 / n * (l0 - l1))
    mx = 1 - math.exp(2 / n * l0)
    r = sdmr2(l1, l0, n)
    assert abs(r.extra["cox_snell"] - cs) < 1e-14
    assert abs(r.statistic - cs / mx) < 1e-14


def test_equal_likelihoods_give_zero():
    assert abs(sdmr2(-12.0, -12.0, 30).statistic) < 1e-15
