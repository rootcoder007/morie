"""Tests for esstst.effective_sample_size_weights (Kish ESS)."""

from morie.fn.esstst import effective_sample_size_weights


def test_esstst_equal_weights_full_n():
    r = effective_sample_size_weights([2.0, 2.0, 2.0, 2.0])
    assert r["ess"] == 4.0
    assert r["efficiency"] == 1.0


def test_esstst_kish_formula_and_concentration():
    # ESS = (sum w)^2 / sum w^2
    w = [10.0, 1.0, 1.0, 1.0]
    r = effective_sample_size_weights(w)
    want = sum(w) ** 2 / sum(v * v for v in w)
    assert abs(r["ess"] - want) < 1e-12
    assert r["ess"] < 4.0
