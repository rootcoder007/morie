"""Tests for morie.fn.zxsgb: a boosted stump recomputed by exhaustive split search."""

import math

from morie.fn.zxsgb import spatial_gbm

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


def test_one_round_stump_is_the_best_single_split():
    r = spatial_gbm(Y, X, S, n_trees=1, rate=1.0, max_depth=1, min_leaf=2)
    m = sum(Y) / N
    res = [v - m for v in Y]
    best = None
    for f in range(3):
        vals = sorted(set(z[f] for z in Z))
        for a, b in zip(vals, vals[1:]):
            t = 0.5 * (a + b)
            L = [res[i] for i in range(N) if Z[i][f] <= t]
            R = [res[i] for i in range(N) if Z[i][f] > t]
            if len(L) < 2 or len(R) < 2:
                continue
            sse = sum((v - sum(L) / len(L)) ** 2 for v in L) + sum((v - sum(R) / len(R)) ** 2 for v in R)
            if best is None or sse < best[0] - 1e-12:
                best = (sse, f, t, sum(L) / len(L), sum(R) / len(R))
    _sse, f, t, ml, mr = best
    for i in range(N):
        assert abs(r.extra["fitted"][i] - (m + (ml if Z[i][f] <= t else mr))) < 1e-12
