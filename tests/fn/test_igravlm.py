"""Tests for morie.fn.igravlm: expected values recomputed from the defining equations."""

import math

from morie.fn.igravlm import igravlm

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


def test_lm_error_on_gravity_residuals():
    W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
    e = [0.3, -0.2, 0.5, -0.4]
    n = 4
    T = sum(W[i][j] * (W[i][j] + W[j][i]) for i in range(n) for j in range(n))
    eWe = sum(e[i] * W[i][j] * e[j] for i in range(n) for j in range(n))
    lm = (n * eWe / sum(v * v for v in e)) ** 2 / T
    r = igravlm(e, W)
    assert abs(r.statistic - lm) < 1e-13
    assert abs(r.p_value - math.erfc(math.sqrt(lm / 2))) < 1e-12
