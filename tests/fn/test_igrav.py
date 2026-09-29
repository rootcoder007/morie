"""Tests for morie.fn.igrav: expected values recomputed from the defining equations."""

import math

from morie.fn.igrav import igrav

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


def test_log_linear_ols_on_positive_flows():
    k = [i for i in range(12) if F[i] > 0]
    X = [[1.0, math.log(MO[i]), math.log(MD[i]), math.log(D[i])] for i in k]
    y = [math.log(F[i]) for i in k]
    b = _ols(X, y)
    r = igrav(F, MO, MD, D)
    assert max(abs(u - v) for u, v in zip(r.extra["coefficients"], b)) < 1e-10
    assert r.statistic == r.extra["coefficients"][3]
    e = [t - sum(u * v for u, v in zip(row, b)) for row, t in zip(X, y)]
    my = sum(y) / len(y)
    assert abs(r.extra["r2"] - (1 - sum(v * v for v in e) / sum((t - my) ** 2 for t in y))) < 1e-12
    assert r.extra["n_used"] == len(k)
