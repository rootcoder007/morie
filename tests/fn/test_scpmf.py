"""Tests for morie.fn.scpmf: impacts of the spatial-lag Poisson model."""

import math

import pytest

from morie.fn._qpcore import solve
from morie.fn.scpmf import scpmf, scpmf_fn

N = 8
W = [[0.5 if abs(i - j) in (1, N - 1) else 0.0 for j in range(N)] for i in range(N)]
X = [
    [1.0, v, w] for v, w in zip((0.3, -0.2, 0.8, 0.1, -0.5, 0.4, 0.9, -0.7), (1.2, 0.4, -0.3, 0.8, 0.0, -1.1, 0.5, 0.2))
]
B = [0.4, 0.6, -0.3]
RHO = 0.35


def _mu(Xm, rho=RHO):
    A = [[(1.0 if i == j else 0.0) - rho * W[i][j] for j in range(N)] for i in range(N)]
    xb = [sum(x * b for x, b in zip(r, B)) for r in Xm]
    return [math.exp(v) for v in solve(A, xb)]


def test_impacts_recomputed_from_inverse():
    A = [[(1.0 if i == j else 0.0) - RHO * W[i][j] for j in range(N)] for i in range(N)]
    cols = [solve(A, [1.0 if i == j else 0.0 for i in range(N)]) for j in range(N)]
    mu = _mu(X)
    d = sum(mu[i] * cols[i][i] for i in range(N)) / N
    t = sum(mu[i] * sum(cols[j][i] for j in range(N)) for i in range(N)) / N
    r = scpmf(B, RHO, X, W)
    for k in range(3):
        assert r.direct[k] == pytest.approx(d * B[k], abs=1e-12)
        assert r.total[k] == pytest.approx(t * B[k], abs=1e-12)
        assert r.indirect[k] == pytest.approx((t - d) * B[k], abs=1e-12)
    assert max(abs(a - b) for a, b in zip(r.fitted, mu)) < 1e-12


def test_impacts_are_mean_derivatives():
    h = 1e-6
    r = scpmf(B, RHO, X, W)
    for k in (1, 2):
        up = [row[:k] + [row[k] + h] + row[k + 1 :] for row in X]
        dn = [row[:k] + [row[k] - h] + row[k + 1 :] for row in X]
        tot = (sum(_mu(up)) - sum(_mu(dn))) / (2 * h) / N
        assert tot == pytest.approx(r.total[k], rel=1e-7)
        dirs = 0.0
        for i in range(N):
            up = [row[:] for row in X]
            dn = [row[:] for row in X]
            up[i][k] += h
            dn[i][k] -= h
            dirs += (_mu(up)[i] - _mu(dn)[i]) / (2 * h)
        assert dirs / N == pytest.approx(r.direct[k], rel=1e-7)


def test_rho_zero_has_no_spillover():
    r = scpmf(B, 0.0, X, W)
    assert max(abs(v) for v in r.indirect) < 1e-15
    assert scpmf_fn is scpmf
    with pytest.raises(ValueError):
        scpmf(B[:2], RHO, X, W)
