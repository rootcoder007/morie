"""Tests for sacmod.spatial_combined (SAC = SARAR with one W)."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.sacmod import spatial_combined


def _data():
    # rook 6 x 6 grid, row-standardised; y from rho = 0.3, lambda = 0.4
    g = 6
    n = g * g
    W = [[0.0] * n for _ in range(n)]
    for i in range(g):
        for j in range(g):
            a = i * g + j
            nb = [(i + di) * g + (j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
                  if 0 <= i + di < g and 0 <= j + dj < g]
            for b in nb:
                W[a][b] = 1.0 / len(nb)
    x = [math.sin(1.7 * k) + 0.3 * math.cos(0.4 * k) for k in range(n)]
    e = [0.5 * math.sin(2.9 * k + 1) for k in range(n)]

    def solve(rho, b):
        v = list(b)
        for _ in range(200):
            v = [b[i] + rho * sum(W[i][j] * v[j] for j in range(n)) for i in range(n)]
        return v
    u = solve(0.4, e)
    y = solve(0.3, [1 + 2 * x[i] + u[i] for i in range(n)])
    return y, [[1.0, v] for v in x], W


def test_sacmod_basic():
    """spatialreg::sacsarlm(y ~ x, type = "sac", method = "eigen") on the
    same data gives rho 0.2981944444567745, lambda -0.6990165795060648,
    beta (1.0153176515621172, 2.0110161363375125), s2 0.0505687419405051
    and log-likelihood -1.130694995295162; its optimiser stops within
    ~2e-7 of the optimum, hence 1e-6 on the parameters, while the flat
    likelihood agrees to 1e-12."""
    y, X, W = _data()
    r = spatial_combined(y, X, W)
    assert r["rho"] == pytest.approx(0.2981944444567745, abs=1e-6)
    assert r["lambda"] == pytest.approx(-0.6990165795060648, abs=1e-6)
    assert list(r["estimate"]) == pytest.approx([1.0153176515621172, 2.0110161363375125], abs=1e-6)
    assert r["loglik"] == pytest.approx(-1.130694995295162, rel=1e-12)
    # sigma2 is the ML value e'e / n at the estimates, recomputed
    n = len(y)
    Wm, Xm, yv = np.array(W), np.array(X), np.array(y)
    A = np.eye(n) - r["rho"] * Wm
    B = np.eye(n) - r["lambda"] * Wm
    res = B @ (A @ yv - Xm @ np.array(list(r["estimate"])))
    assert r["sigma2"] == pytest.approx(float(res @ res) / n, rel=1e-12)


def test_sacmod_edge():
    """Standard errors come from the full information in (beta, rho,
    lambda, sigma2): uncertainty in rho and lambda can only widen them
    beyond the conditional sigma2 (X*'X*)^-1 values (a marginal
    variance is never below a conditional one)."""
    y, X, W = _data()
    r = spatial_combined(y, X, W)
    n = len(y)
    Wm, Xm = np.array(W), np.array(X)
    B = np.eye(n) - r["lambda"] * Wm
    Xs = B @ Xm
    cond = np.linalg.inv(Xs.T @ Xs).tolist()
    for j in range(2):
        assert r["se"][j] >= math.sqrt(r["sigma2"] * cond[j][j]) * (1 - 1e-9)
    assert r["se_rho"] > 0 and r["se_lambda"] > 0
    with pytest.raises(ValueError):
        spatial_combined(y[:-1], X, W)
