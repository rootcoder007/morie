"""Tests for semmod.spatial_error_model."""

import math

import pytest

from morie.fn.semmod import spatial_error_model


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


def _ll_err(lam):
    ys = [a - lam * b for a, b in zip(Y, _lag(Y))]
    cols = [[r[j] for r in XI] for j in range(3)]
    lc = [_lag(c) for c in cols]
    xs = [[XI[i][j] - lam * lc[j][i] for j in range(3)] for i in range(N)]
    b = _ols(xs, ys)
    sse = sum((v - sum(c * x for c, x in zip(b, r))) ** 2 for v, r in zip(ys, xs))
    return _logdet(lam) - N / 2 * (math.log(2 * math.pi * sse / N) + 1), b


def test_semmod_basic():
    """The maximised log-likelihood equals spatialreg 1.3 errorsarlm(method =
    "eigen") on the same data: -24.910016717573413."""
    r = spatial_error_model(Y, X, W)
    assert r["loglik"] == pytest.approx(-24.910016717573413, rel=1e-12)
    assert r["loglik"] == pytest.approx(_ll_err(r["lambda"])[0], rel=1e-12)


def test_semmod_edge():
    """lambda is a profile maximum and beta is GLS at that lambda."""
    r = spatial_error_model(Y, X, W)
    lam = r["lambda"]
    for h in (1e-3, 1e-4):
        assert _ll_err(lam - h)[0] <= r["loglik"] and _ll_err(lam + h)[0] <= r["loglik"]
    assert r["beta"] == pytest.approx(_ll_err(lam)[1], rel=1e-10)
    with pytest.raises(ValueError, match="W must be"):
        spatial_error_model(Y, X, [1.0] * N)


