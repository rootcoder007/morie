"""Tests for morie.fn.slxboot: every expected value is recomputed from the formula."""

import math

from morie.fn._rng import random_uniform
from morie.fn.slxboot import slxboot


def _inv(A):
    n = len(A)
    M = [list(map(float, r)) + [1.0 if i == j else 0.0 for j in range(n)] for i, r in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(n):
            if r != c:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [r[n:] for r in M]


def _mm(A, B):
    return [[sum(a * b for a, b in zip(r, c)) for c in zip(*B)] for r in A]


def _mv(A, v):
    return [sum(a * b for a, b in zip(r, v)) for r in A]


def _t(A):
    return [list(c) for c in zip(*A)]


def _imw(W, a):
    return [[(1.0 if i == j else 0.0) - a * W[i][j] for j in range(len(W))] for i in range(len(W))]


def _det(A):
    # Laplace expansion (small matrices only)
    if len(A) == 1:
        return A[0][0]
    return sum((-1) ** j * A[0][j] * _det([r[:j] + r[j + 1 :] for r in A[1:]]) for j in range(len(A)))


N = 8
_B = [[1.0 if abs(i - j) == 1 or {i, j} == {0, 7} or {i, j} == {2, 5} else 0.0 for j in range(N)] for i in range(N)]
W = [[v / sum(r) for v in r] for r in _B]
X = [[1.0, ((i * 7) % 11) / 5] for i in range(N)]
Y = [1 + 2 * X[i][1] + ((i * 3) % 5 - 2) / 4 + 0.3 * sum(W[i][j] * X[j][1] for j in range(N)) for i in range(N)]
E = [((i * 4) % 7 - 3) / 5 for i in range(N)]
W3 = [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]


def _aslist(v):
    return v.tolist() if hasattr(v, "tolist") else list(v)


def _ols2(Z, v):
    # normal equations for 3 columns by Cramer's rule
    A = [[sum(r[a] * r[b] for r in Z) for b in range(3)] for a in range(3)]
    c = [sum(r[a] * t for r, t in zip(Z, v)) for a in range(3)]
    d = _det(A)
    return [_det([[c[i] if j == k else A[i][j] for j in range(3)] for i in range(3)]) / d for k in range(3)]


def _q7(x, q):
    s = sorted(x)
    h = (len(s) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def test_bootstrap_of_theta():
    Z = [r + [sum(W[i][j] * X[j][1] for j in range(N))] for i, r in enumerate(X)]
    g = _ols2(Z, Y)
    fit = _mv(Z, g)
    e = [a - b for a, b in zip(Y, fit)]
    m = sum(e) / N
    e = [v - m for v in e]
    B = 30
    u = [float(v) for v in _aslist(random_uniform(B * N, seed=4))]
    th = [_ols2(Z, [fit[i] + e[int(u[b * N + i] * N)] for i in range(N)])[2] for b in range(B)]
    r = slxboot(Y, X, W, B=B, seed=4)
    assert abs(r.statistic - g[2]) < 1e-11
    assert abs(r.extra["ci_lower"][0] - _q7(th, 0.025)) < 1e-10
    assert abs(r.extra["ci_upper"][0] - _q7(th, 0.975)) < 1e-10
    assert r.extra["lagged_columns"] == [1]
