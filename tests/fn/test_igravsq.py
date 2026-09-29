"""Tests for morie.fn.igravsq: expected values recomputed from the defining equations."""

import math

from morie.fn.igravsq import igravsq

F = [12.0, 0.0, 30.0, 2.0, 85.0, 4.0, 19.0, 1.0, 7.0, 40.0, 3.0, 16.0]
MO = [5.0, 5, 9, 9, 20, 20, 7, 7, 12, 12, 3, 3]
MD = [9.0, 20, 5, 20, 5, 9, 20, 9, 7, 3, 12, 5]
D = [1.0, 3.0, 1.0, 2.0, 3.0, 2.0, 2.5, 1.5, 2.0, 1.0, 3.0, 1.2]


def _solve(A, b):
    n = len(A)
    M = [list(r) + [v] for r, v in zip(A, b)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * q for a, q in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def _ols(X, y):
    p = len(X[0])
    return _solve(
        [[sum(r[a] * r[b] for r in X) for b in range(p)] for a in range(p)],
        [sum(r[a] * t for r, t in zip(X, y)) for a in range(p)],
    )


def test_symmetric_square_gravity():
    Fm = [[0, 12, 5, 3], [10, 0, 8, 2], [6, 9, 0, 7], [2, 3, 8, 0]]
    Dm = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    M = [10.0, 8.0, 12.0, 5.0]
    X, y = [], []
    for i in range(4):
        for j in range(4):
            if i != j:
                X.append([1.0, math.log(M[i] * M[j]), math.log(Dm[i][j])])
                y.append(math.log(Fm[i][j]))
    b = _ols(X, y)
    r = igravsq(Fm, M, Dm)
    assert max(abs(u - v) for u, v in zip(r.extra["coefficients"], b)) < 1e-10
    num = sum(abs(Fm[i][j] - Fm[j][i]) for i in range(4) for j in range(i + 1, 4))
    den = sum(Fm[i][j] + Fm[j][i] for i in range(4) for j in range(i + 1, 4))
    assert abs(r.extra["asymmetry"] - num / den) < 1e-15
