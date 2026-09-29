"""Tests for morie.fn.igravcl: expected values recomputed from the defining equations."""

import math

from morie.fn.igravcl import igravcl

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


def test_poisson_score_equations_match_total_and_mean_log_distance():
    r = igravcl(F, MO, MD, D)
    beta, k = r.statistic, r.extra["k"]
    fit = [k * a * b * d ** (-beta) for a, b, d in zip(MO, MD, D)]
    assert max(abs(u - v) for u, v in zip(fit, r.extra["fitted"])) < 1e-9
    assert abs(sum(fit) - sum(F)) < 1e-9
    assert abs(sum(f * math.log(d) for f, d in zip(fit, D)) - sum(f * math.log(d) for f, d in zip(F, D))) < 1e-9
