"""Tests for svdpp (SVD++, Koren 2008 eq. 15)."""

import math

import pytest

from morie.fn.svdpp import fit_svdpp, implicit_term, predict, sgd_step


def test_svdpp_basic():
    """One SGD step follows Koren (2008): with e = r - rhat,
    b += g(e - l b), q_i += g(e (p_u + |N|^-1/2 sum y) - l q_i),
    p_u += g(e q_i - l p_u), y_j += g(e |N|^-1/2 q_i - l y_j), and the
    prediction puts the implicit term inside the inner product."""
    y = {0: [0.1, -0.2], 1: [0.3, 0.05], 2: [-0.1, 0.2]}
    N = [0, 2]
    p, q = [0.2, 0.1], [0.4, -0.3]
    imp = [(y[0][t] + y[2][t]) / math.sqrt(2) for t in range(2)]
    eff = [p[t] + imp[t] for t in range(2)]
    rhat = 3.5 + 0.2 - 0.1 + sum(q[t] * eff[t] for t in range(2))
    assert predict(3.5, 0.2, -0.1, p, q, N, y)["prediction"] == pytest.approx(rhat, abs=1e-15)
    g, lam = 0.01, 0.02
    e = 4.0 - rhat
    st = sgd_step(4.0, 3.5, 0.2, -0.1, p, q, N, y, lr=g, reg=lam)
    assert st["error"] == pytest.approx(e, abs=1e-15)
    assert st["b_user"] == pytest.approx(0.2 + g * (e - lam * 0.2), abs=1e-15)
    assert st["q_i"] == pytest.approx([q[t] + g * (e * eff[t] - lam * q[t]) for t in range(2)], abs=1e-15)
    assert st["p_u"] == pytest.approx([p[t] + g * (e * q[t] - lam * p[t]) for t in range(2)], abs=1e-15)
    for j in N:
        assert st["y"][j] == pytest.approx([y[j][t] + g * (e * q[t] / math.sqrt(2) - lam * y[j][t]) for t in range(2)], abs=1e-15)
    assert set(st["y"]) == set(N)


def test_svdpp_edge():
    """The implicit term scales by |N|^exponent; training error falls;
    implicit=False fits plain SVD; no ratings raise."""
    y = {0: [1.0], 1: [3.0]}
    assert implicit_term([0, 1], y, exponent=-1.0)["term"] == [2.0]
    assert implicit_term([0, 1], y, exponent=0.0)["term"] == [4.0]
    R = [(u, i, 1.0 + ((3 * u + 5 * i) % 5)) for u in range(6) for i in range(5) if (u + i) % 3]
    f = fit_svdpp(R, 6, 5, factors=2, epochs=40, lr=0.02, seed=1)
    assert f["rmse_history"][-1] < f["rmse_history"][0]
    assert fit_svdpp(R, 6, 5, factors=2, epochs=3, implicit=False)["Y"] is None
    assert f["mu"] == pytest.approx(sum(r for _, _, r in R) / len(R), abs=1e-15)
    with pytest.raises(ValueError):
        fit_svdpp([], 1, 1)
