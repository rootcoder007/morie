"""Tests for sarmod.spatial_lag_model."""

import math

import pytest

from morie.fn.sarmod import spatial_lag_model


N, NC = 20, 4
W = [[1.0 if abs(i // NC - j // NC) + abs(i % NC - j % NC) == 1 else 0.0 for j in range(N)]
     for i in range(N)]
W = [[v / sum(r) for v in r] for r in W]          # row-standardised rook grid
X1 = [float((i * 5) % 9) for i in range(N)]
X2 = [math.cos(i) for i in range(N)]
Y = [1 + 0.8 * a - 0.5 * b + ((i * 13) % 7 - 3) / 2 + 0.3 * (i // NC)
     for i, (a, b) in enumerate(zip(X1, X2))]
X = [[a, b] for a, b in zip(X1, X2)]
XI = [[1.0, a, b] for a, b in zip(X1, X2)]


def _logdet(rho):
    A = [[(1.0 if i == j else 0.0) - rho * W[i][j] for j in range(N)] for i in range(N)]
    s = 0.0
    for c in range(N):
        p = max(range(c, N), key=lambda r: abs(A[r][c]))
        A[c], A[p] = A[p], A[c]
        s += math.log(abs(A[c][c]))
        for r in range(c + 1, N):
            f = A[r][c] / A[c][c]
            A[r] = [A[r][k] - f * A[c][k] for k in range(N)]
    return s


def _ols(Xm, y):
    k = len(Xm[0])
    G = [[sum(r[a] * r[b] for r in Xm) for b in range(k)] + [sum(r[a] * v for r, v in zip(Xm, y))]
         for a in range(k)]
    for c in range(k):
        for r in range(k):
            if r != c:
                f = G[r][c] / G[c][c]
                G[r] = [G[r][t] - f * G[c][t] for t in range(k + 1)]
    return [G[a][k] / G[a][a] for a in range(k)]


def _lag(v):
    return [sum(W[i][j] * v[j] for j in range(N)) for i in range(N)]


def _ll_lag(rho):
    ys = [a - rho * b for a, b in zip(Y, _lag(Y))]
    b = _ols(XI, ys)
    sse = sum((v - sum(c * x for c, x in zip(b, r))) ** 2 for v, r in zip(ys, XI))
    return _logdet(rho) - N / 2 * (math.log(2 * math.pi * sse / N) + 1)


def test_sarmod_basic():
    """The maximised log-likelihood equals spatialreg 1.3 lagsarlm(method =
    "eigen") on the same data: -25.372816101836037."""
    r = spatial_lag_model(Y, X, W)
    assert r["loglik"] == pytest.approx(-25.372816101836037, rel=1e-12)
    assert r["loglik"] == pytest.approx(_ll_lag(r["rho"]), rel=1e-12)


def test_sarmod_edge():
    """rho is a profile maximum and beta is OLS of (I - rho W) y on X."""
    r = spatial_lag_model(Y, X, W)
    rho = r["rho"]
    for h in (1e-3, 1e-4):
        assert _ll_lag(rho - h) <= r["loglik"] and _ll_lag(rho + h) <= r["loglik"]
    b = _ols(XI, [a - rho * c for a, c in zip(Y, _lag(Y))])
    assert r["beta"] == pytest.approx(b, rel=1e-10)
    with pytest.raises(ValueError, match="W must be"):
        spatial_lag_model(Y, X, [1.0] * N)


