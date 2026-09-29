"""Tests for morie.fn.zxsqr: check-loss optimality of the quantile fit."""

import math

from morie.fn.zxsqr import spatial_quantile

N = 24
S = [[(i % 6) / 5 + ((i * 7) % 11) / 50, (i // 6) / 4 + ((i * 3) % 7) / 40] for i in range(N)]
X = [[((i * 7) % 11) / 10] for i in range(N)]
Y = [1 + 2 * x[0] + math.sin(3 * s[0]) + s[1] ** 2 + ((i * 3) % 5 - 2) / 10 for i, (x, s) in enumerate(zip(X, S))]


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


def _ols(rows, y):
    D = [[1.0] + r for r in rows]
    p = len(D[0])
    return _solve(
        [[sum(r[a] * r[b] for r in D) for b in range(p)] for a in range(p)],
        [sum(r[a] * t for r, t in zip(D, y)) for a in range(p)],
    )


def _pred(b, r):
    return b[0] + sum(u * v for u, v in zip(b[1:], r))


Z = [x + s for x, s in zip(X, S)]


def _loss(b, tau):
    return sum((tau if e >= 0 else tau - 1) * e for e in (Y[i] - _pred(b, Z[i]) for i in range(N)))


def test_objective_and_local_optimality():
    for tau in (0.3, 0.5):
        r = spatial_quantile(Y, X, S, tau)
        b = r.extra["coefficients"]
        assert abs(r.value - _loss(b, tau)) < 1e-10
        for k in range(len(b)):
            for d in (1e-4, -1e-4):
                bb = list(b)
                bb[k] += d
                assert _loss(bb, tau) >= r.value - 1e-12
