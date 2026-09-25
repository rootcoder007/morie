"""Tests for straprp.stratified_proportion (Cochran 1977, sec. 5.5)."""

import math

import pytest

from morie.fn.straprp import stratified_proportion


def _data():
    y = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0]
    h = [1] * 6 + [2] * 5 + [3] * 7
    return y, h


def test_straprp_basic():
    """p_st = sum W_h p_h; with known N_h the variance carries the
    finite population correction, sum W_h^2 (1 - n_h/N_h) p_h q_h/(n_h - 1)
    (eq. 5.47).  samplingbook::stratamean(y, h, Nh = c(40, 25, 60))
    reports the same mean 0.5962 and standard error 0.1099."""
    y, h = _data()
    Nh = [40, 25, 60]
    ph = [4 / 6, 1 / 5, 5 / 7]
    nh = [6, 5, 7]
    W = [v / 125 for v in Nh]
    p = sum(w * q for w, q in zip(W, ph))
    var = sum(w * w * (1 - n / N) * q * (1 - q) / (n - 1) for w, n, N, q in zip(W, nh, Nh, ph))
    r = stratified_proportion(y, h, N_h=Nh)
    assert r["proportion"] == pytest.approx(p, abs=1e-15)
    assert r["variance"] == pytest.approx(var, rel=1e-13)
    assert r["se"] == pytest.approx(math.sqrt(var), rel=1e-13)
    assert round(r["proportion"], 4) == 0.5962 and round(r["se"], 4) == 0.1099
    assert r["weights_are_population_shares"] is True


def test_straprp_edge():
    """Population shares without sizes get no fpc; sample shares
    reproduce the unweighted mean; bad inputs raise."""
    y, h = _data()
    W = [0.32, 0.2, 0.48]
    ph = [4 / 6, 1 / 5, 5 / 7]
    r = stratified_proportion(y, h, weights=W)
    assert r["variance"] == pytest.approx(sum(w * w * q * (1 - q) / (n - 1) for w, q, n in zip(W, ph, [6, 5, 7])), rel=1e-13)
    r0 = stratified_proportion(y, h)
    assert r0["proportion"] == pytest.approx(sum(y) / len(y), abs=1e-15)
    assert r0["weights_are_population_shares"] is False
    with pytest.raises(ValueError):
        stratified_proportion([2] + y[1:], h)
    with pytest.raises(ValueError):
        stratified_proportion(y, h, weights=[0.5, 0.2, 0.2])
    with pytest.raises(ValueError):
        stratified_proportion(y, h, N_h=[4, 25, 60])
    with pytest.raises(ValueError):
        stratified_proportion(y, [1] * len(y))
