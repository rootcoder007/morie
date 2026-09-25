"""Tests for timeRS (time-dependent biases, Koren 2010)."""

import math

import pytest

from morie.fn.timeRS import deviation, fit_time_bias, predict_time, time_bin


R = [(0, 0, 10.0, 4.0), (0, 1, 150.0, 3.0), (1, 0, 20.0, 5.0), (1, 2, 400.0, 2.0),
     (2, 1, 90.0, 3.5), (2, 2, 95.0, 4.5), (0, 2, 300.0, 3.0)]


def _replay(epochs, lr, reg, beta=0.4, bd=70, nb=30):
    """Koren (2010) eq. (8) biases, updated rating by rating:
    e = r - (mu + b_u + a_u dev_u(t) + b_i + b_{i,Bin(t)}), then each
    parameter moves by lr (e * dpred/dparam - reg * param)."""
    mu = sum(r for *_, r in R) / len(R)
    days = {}
    for u, _, t, _ in R:
        days.setdefault(u, []).append(t)
    tu = {u: sum(v) / len(v) for u, v in days.items()}
    bu, al, bi = [0.0] * 3, [0.0] * 3, [0.0] * 3
    bins = [[0.0] * nb for _ in range(3)]
    hist = []
    for _ in range(epochs):
        se = 0.0
        for u, i, t, r in R:
            d = t - tu[u]
            dev = math.copysign(abs(d) ** beta, d) if d else 0.0
            j = min(int(t // bd), nb - 1)
            e = r - (mu + bu[u] + al[u] * dev + bi[i] + bins[i][j])
            se += e * e
            bu[u] += lr * (e - reg * bu[u])
            al[u] += lr * (e * dev - reg * al[u])
            bi[i] += lr * (e - reg * bi[i])
            bins[i][j] += lr * (e - reg * bins[i][j])
        hist.append(math.sqrt(se / len(R)))
    return bu, al, bi, bins, hist


def test_timeRS_basic():
    """Every fitted parameter and the RMSE path equal an independent
    replay of the stochastic-gradient updates."""
    bu, al, bi, bins, hist = _replay(5, 0.005, 0.02)
    f = fit_time_bias(R, 3, 3, epochs=5, lr=0.005, reg=0.02)
    assert f["b_user"] == pytest.approx(bu, abs=1e-14)
    assert f["alpha_user"] == pytest.approx(al, abs=1e-14)
    assert f["b_item"] == pytest.approx(bi, abs=1e-14)
    for got, ref in zip(f["item_bins"], bins):
        assert got == pytest.approx(ref, abs=1e-14)
    assert f["rmse_history"] == pytest.approx(hist, abs=1e-14)


def test_timeRS_edge():
    """dev is signed and concave; bins saturate at the last one; the
    prediction adds mu, both biases and the factor inner product."""
    assert deviation(116.0, 16.0) == pytest.approx(100 ** 0.4, rel=1e-15)
    assert deviation(0.0, 100.0) == pytest.approx(-(100 ** 0.4), rel=1e-15)
    assert deviation(5.0, 5.0) == 0.0
    assert time_bin(10_000, 70, 30) == 29
    pr = predict_time(3.0, 0.2, 0.1, 50.0, -0.1, [0.0, 0.05], 100.0, [1.0, 2.0], [0.5, 0.25])
    assert pr["prediction"] == pytest.approx(3.0 + 0.2 + 0.1 * 50 ** 0.4 - 0.1 + 0.05 + 1.0, rel=1e-14)
    with pytest.raises(ValueError):
        deviation(1.0, 0.0, beta=0.0)
    with pytest.raises(ValueError):
        fit_time_bias([], 1, 1)
