"""Tests for morie.fn.igravfe: expected values recomputed from the defining equations."""

import math

from morie.fn.igravfe import igravfe

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


ORIG = ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
DE = ["x", "y", "z", "x", "y", "z", "x", "y", "z", "x", "y", "z"]


def test_fixed_effects_reproduce_origin_and_destination_totals():
    r = igravfe(F, ORIG, DE, D)
    mu = r.extra["fitted"]
    for lab, ids in (("o", ORIG), ("d", DE)):
        for v in set(ids):
            s_obs = sum(f for f, g in zip(F, ids) if g == v)
            s_fit = sum(m for m, g in zip(mu, ids) if g == v)
            assert abs(s_obs - s_fit) < 1e-8, (lab, v)
    assert abs(sum(m * math.log(d) for m, d in zip(mu, D)) - sum(f * math.log(d) for f, d in zip(F, D))) < 1e-8
    a, g = r.extra["origin_effects"], r.extra["destination_effects"]
    for i in range(12):
        assert abs(mu[i] - math.exp(a[ORIG[i]] + g[DE[i]] + r.statistic * math.log(D[i]))) < 1e-9 * max(1.0, mu[i])
    assert g["x"] == 0.0
