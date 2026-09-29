"""Tests for morie.fn.igravnb: expected values recomputed from the defining equations."""

import math

from morie.fn.igravnb import igravnb

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


def _ll(beta, theta):
    s = 0.0
    for f, a, b, d in zip(F, MO, MD, D):
        mu = math.exp(beta[0] + beta[1] * math.log(a) + beta[2] * math.log(b) + beta[3] * math.log(d))
        s += (
            math.lgamma(theta + f)
            - math.lgamma(theta)
            - math.lgamma(f + 1)
            + theta * math.log(theta)
            + (f * math.log(mu) if f > 0 else 0.0)
            - (theta + f) * math.log(theta + mu)
        )
    return s


def test_nb2_score_equations():
    r = igravnb(F, MO, MD, D)
    b, th = r.extra["coefficients"], r.extra["theta"]
    mu = r.extra["fitted"]
    X = [[1.0, math.log(a), math.log(c), math.log(d)] for a, c, d in zip(MO, MD, D)]
    for k in range(4):
        s = sum(x[k] * (f - m) / (1 + m / th) for x, f, m in zip(X, F, mu))
        assert abs(s) < 1e-8
    assert abs(r.extra["loglik"] - _ll(b, th)) < 1e-10
    h = 1e-5 * th
    assert abs((_ll(b, th + h) - _ll(b, th - h)) / (2 * h)) < 1e-6
