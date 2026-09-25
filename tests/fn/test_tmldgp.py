"""Tests for tmldgp (post-lasso nuisances, unpenalised targeting)."""

import math
import statistics

import pytest

from morie.fn.tmldgp import (lasso_path, penalised_tmle, post_lasso,
                             shrunk_targeting_unsafe)


def _expit(x):
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p):
    return math.log(p / (1.0 - p))


def _data(n=60, p=4):
    X = [[math.sin(1.1 * k + j) for j in range(p)] for k in range(n)]
    A = [1.0 if ((41 * k + 9) % 89 + 0.5) / 89.0 < _expit(0.8 * x[0]) else 0.0 for k, x in enumerate(X)]
    Y = [min(max(0.3 + 0.2 * a + 0.25 * x[1] + 0.02 * ((7 * k) % 5 - 2), 0.0), 1.0)
         for k, (a, x) in enumerate(zip(A, X))]
    return Y, A, X


def _kkt(X, y, fit, lam):
    """Lasso optimality for (1/2n)|y - b0 - Xb|^2 + lam|b|_1: the
    correlation of each column with the residual is lam*sign(b_j) on the
    support and at most lam off it; the intercept makes residuals sum to 0."""
    n = len(y)
    r = [y[i] - fit["intercept"] - sum(a * b for a, b in zip(X[i], fit["beta"])) for i in range(n)]
    assert sum(r) == pytest.approx(0.0, abs=1e-9)
    for j, b in enumerate(fit["beta"]):
        c = sum(X[i][j] * r[i] for i in range(n)) / n
        if abs(b) > 1e-10:
            assert c == pytest.approx(lam * math.copysign(1.0, b), abs=1e-7)
        else:
            assert abs(c) <= lam + 1e-7


def test_tmldgp_basic():
    """Lasso satisfies its KKT conditions, post-lasso residuals are
    orthogonal to the selected columns, and the TMLE equals a bisection
    solve of the unpenalised logistic score on the post-lasso fits."""
    Y, A, X = _data()
    lam = 0.05
    fit = lasso_path(X, Y, lam, iters=5000, tol=1e-13)
    # tol 1e-7 on KKT: coordinate descent stops when no coefficient moves
    # by more than 1e-13, and the columns have norm^2/n <= 1, so the
    # residual correlations are exact to far better than 1e-7
    _kkt(X, Y, fit, lam)
    pl = post_lasso(X, Y, lam)
    S = pl["support"]
    res = [Y[i] - pl["predict"](X[i]) for i in range(len(Y))]
    assert sum(res) == pytest.approx(0.0, abs=1e-9)
    for j in S:
        assert sum(X[i][j] * res[i] for i in range(len(Y))) == pytest.approx(0.0, abs=1e-9)

    n = len(Y)
    g = [min(max(post_lasso(X, A, lam)["predict"](X[i]), 0.02), 0.98) for i in range(n)]
    qf = post_lasso([[a] + x for a, x in zip(A, X)], Y, lam)
    q1 = [min(max(qf["predict"]([1.0] + X[i]), 1e-6), 1 - 1e-6) for i in range(n)]
    q0 = [min(max(qf["predict"]([0.0] + X[i]), 1e-6), 1 - 1e-6) for i in range(n)]
    H = [a / p - (1 - a) / (1 - p) for a, p in zip(A, g)]
    qa = [x if a else z for a, x, z in zip(A, q1, q0)]
    lo, hi = -50.0, 50.0
    for _ in range(200):
        m = 0.5 * (lo + hi)
        s = sum(h * (y - _expit(_logit(q) + m * h)) for h, y, q in zip(H, Y, qa))
        lo, hi = (m, hi) if s > 0 else (lo, m)
    e = 0.5 * (lo + hi)
    q1s = [_expit(_logit(q) + e / p) for q, p in zip(q1, g)]
    q0s = [_expit(_logit(q) - e / (1 - p)) for q, p in zip(q0, g)]
    psi = statistics.fmean(x - z for x, z in zip(q1s, q0s))
    d = [h * (y - (x if a else z)) + x - z - psi for h, y, a, x, z in zip(H, Y, A, q1s, q0s)]
    r = penalised_tmle(Y, A, X, penalty=lam)
    assert r["estimate"] == pytest.approx(psi, abs=1e-9)
    assert r["epsilon"] == pytest.approx(e, abs=1e-9)
    assert r["se"] == pytest.approx(statistics.pstdev(d) / math.sqrt(n), rel=1e-8)
    assert r["solves_eic"]


def test_tmldgp_edge():
    """Out-of-range outcome and negative lambda raise; an enormous penalty
    selects nothing; a ridge on epsilon leaves the score unsolved."""
    Y, A, X = _data()
    with pytest.raises(ValueError):
        penalised_tmle([1.5] + Y[1:], A, X)
    with pytest.raises(ValueError):
        lasso_path(X, Y, -1.0)
    assert post_lasso(X, Y, 1e6)["support"] == []
    Q = [0.4] * 10
    Hh = [1.0, -1.0] * 5
    Yy = [1.0, 0.0] * 5
    assert abs(shrunk_targeting_unsafe(Q, Hh, Yy, ridge=5.0)["score"]) > 1e-3
    assert abs(shrunk_targeting_unsafe(Q, Hh, Yy, ridge=0.0)["score"]) < 1e-9
