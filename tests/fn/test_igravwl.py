"""Tests for morie.fn.igravwl: expected values recomputed from the defining equations."""

import math

from morie.fn.igravwl import igravwl

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


def test_doubly_constrained_structure():
    og, Dt = [10.0, 20.0, 15.0], [18.0, 12.0, 15.0]
    C = [[1.0, 2.0, 3.0], [2.0, 1.0, 2.5], [3.0, 2.0, 1.0]]
    beta = 0.7
    obs = [[5.0, 3.0, 2.0], [6.0, 9.0, 5.0], [7.0, 0.0, 8.0]]
    r = igravwl(obs, og, Dt, C, beta)
    T = r.extra["T"]
    for i in range(3):
        assert abs(sum(T[i]) - og[i]) < 1e-9
        assert abs(sum(T[k][i] for k in range(3)) - Dt[i]) < 1e-9
    # log T_ij + beta c_ij is additive in i and j
    got = T[0][0] * T[1][1] / (T[0][1] * T[1][0])
    assert abs(got - math.exp(-beta * (C[0][0] + C[1][1] - C[0][1] - C[1][0]))) < 1e-10
    mc = sum(T[i][j] * C[i][j] for i in range(3) for j in range(3)) / sum(map(sum, T))
    assert abs(r.statistic - mc) < 1e-12
    rmse = math.sqrt(sum((obs[i][j] - T[i][j]) ** 2 for i in range(3) for j in range(3)) / 9)
    assert abs(r.extra["srmse"] - rmse / (sum(map(sum, obs)) / 9)) < 1e-12
