"""Tests for morie.fn.sarsim: every expected value is recomputed from the formula."""

import math

from morie.fn._rng import random_normal
from morie.fn.sarsim import sarsim


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


def _impacts(beta, rho, Wm):
    n = len(Wm)
    Ai = _inv(_imw(Wm, rho))
    tr, tot = sum(Ai[i][i] for i in range(n)), sum(map(sum, Ai))
    return [b * tr / n for b in beta], [b * tot / n for b in beta]


def _aslist(v):
    return v.tolist() if hasattr(v, "tolist") else list(v)


def test_draws_are_philox_normals_through_the_cholesky_factor():
    V = [[0.01, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.09]]
    ns = 12
    z = [float(v) for v in _aslist(random_normal(ns * 3, seed=5))]
    tot0, tot1, dirs = [], [], []
    for s in range(ns):
        rho = 0.4 + 0.1 * z[3 * s]
        b = [2.0 + 0.2 * z[3 * s + 1], -1.0 + 0.3 * z[3 * s + 2]]
        d, t = _impacts(b, rho, W)
        tot0.append(t[0])
        tot1.append(t[1])
        dirs.append(d[0])
    r = sarsim([2.0, -1.0], 0.4, W, nsim=ns, vcov=V, seed=5)
    assert abs(r.statistic - sum(tot0) / ns) < 1e-12
    assert abs(r.extra["mean"]["total"][1] - sum(tot1) / ns) < 1e-12
    m = sum(dirs) / ns
    assert abs(r.extra["sd"]["direct"][0] - math.sqrt(sum((v - m) ** 2 for v in dirs) / (ns - 1))) < 1e-12
    d, t = _impacts([2.0, -1.0], 0.4, W)
    assert abs(r.extra["point"]["total"][0] - t[0]) < 1e-12


def test_vcov_required():
    import pytest

    with pytest.raises(ValueError):
        sarsim([2.0], 0.4, W3)
