"""Tests for morie.fn.mgwrdg."""

import math

from morie.fn.mgwrdg import mgwrdg


def test_criteria():
    ll, tr, n = -31.5, 7.25, 50
    r = mgwrdg(ll, tr, n, k=4)
    assert abs(r["AIC"] - (63.0 + 2 * 8.25)) < 1e-12
    assert abs(r["BIC"] - (63.0 + 8.25 * math.log(50))) < 1e-12
    assert abs(r["AICc"] - (63.0 - 50 + 50 * 57.25 / 40.75)) < 1e-12
    assert abs(r["alpha_adjusted"] - 0.05 * 4 / 7.25) < 1e-15
    assert r["df_residual"] == 42.75
