"""Tests for grbag (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grbag as mod


def test_grbag_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_bagging_aggregate_and_disagreement_recomputed():
    import pytest

    P = [[1.0, 2.0, 0.5], [3.0, 4.0, 0.7], [2.0, 2.5, 0.6], [1.5, 3.5, 0.2]]
    B, m = 4, 3
    mean = [sum(P[b][j] for b in range(B)) / B for j in range(m)]
    var = [sum((P[b][j] - mean[j]) ** 2 for b in range(B)) / (B - 1) for j in range(m)]
    r = mod.geron_bagging_predictor(P)
    assert r["prediction"] == pytest.approx(mean, rel=1e-14)
    assert r["per_instance_variance"] == pytest.approx(var, rel=1e-13)
    assert r["se"] == pytest.approx((sum(var) / m / B) ** 0.5, rel=1e-13)
