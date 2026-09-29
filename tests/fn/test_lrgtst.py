"""Tests for lrgtst.g_test (likelihood-ratio G test)."""

import math

import pytest

from morie.fn.lrgtst import g_test


def test_goodness_of_fit_g_recomputed():
    o = [18.0, 30.0, 12.0]
    p = [0.25, 0.5, 0.25]
    e = [60 * v for v in p]
    g = 2 * sum(a * math.log(a / b) for a, b in zip(o, e))
    r = g_test(o, expected=p)
    assert r["statistic"] == pytest.approx(g, rel=1e-13)
    assert r["df"] == 2
    assert r["p_value"] == pytest.approx(math.exp(-g / 2), rel=1e-10)  # chi2(2) survival


def test_independence_g_on_a_2x3_table():
    t = [[10.0, 20.0, 30.0], [20.0, 15.0, 5.0]]
    n = 100.0
    rs = [60.0, 40.0]
    cs = [30.0, 35.0, 35.0]
    g = 2 * sum(t[i][j] * math.log(t[i][j] / (rs[i] * cs[j] / n)) for i in range(2) for j in range(3))
    r = g_test(t)
    assert r["statistic"] == pytest.approx(g, rel=1e-13)
    assert r["df"] == 2
