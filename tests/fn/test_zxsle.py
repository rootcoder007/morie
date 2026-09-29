"""Tests for morie.fn.zxsle: elastic-net Karush-Kuhn-Tucker conditions checked."""

import math

from morie.fn.zxsle import spatial_elastic

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


def _kkt(coef, lam, alpha, trend_rows):
    n, p = len(Y), len(trend_rows[0])
    cols = [[r[j] for r in trend_rows] for j in range(p)]
    mu = [sum(c) / n for c in cols]
    sd = [math.sqrt(sum((v - m) ** 2 for v in c) / n) for c, m in zip(cols, mu)]
    bs = [b * s for b, s in zip(coef[1:], sd)]
    res = [Y[i] - coef[0] - sum(b * v for b, v in zip(coef[1:], trend_rows[i])) for i in range(n)]
    for j in range(p):
        g = sum(((trend_rows[i][j] - mu[j]) / sd[j]) * res[i] for i in range(n)) / n
        if bs[j] != 0.0:
            assert abs(g - lam * alpha * math.copysign(1.0, bs[j]) - lam * (1 - alpha) * bs[j]) < 1e-8
        else:
            assert abs(g) <= lam * alpha + 1e-8


def _quad():
    return [x + [s[0], s[1], s[0] ** 2, s[0] * s[1], s[1] ** 2] for x, s in zip(X, S)]


def test_elastic_net_kkt():
    r = spatial_elastic(Y, X, S, 0.05, 0.3)
    _kkt(r.extra["coefficients"], 0.05, 0.3, _quad())
