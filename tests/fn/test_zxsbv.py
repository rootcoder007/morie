"""Tests for morie.fn.zxsbv: block folds and out-of-fold predictions recomputed."""

import math

from morie.fn.zxsbv import spatial_cv_block

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


def test_blocks_stay_together_and_predictions_are_out_of_fold():
    r = spatial_cv_block(Y, X, S, n_blocks=3, k=3, seed=5)
    folds = r.extra["folds"]
    xs, ys = [s[0] for s in S], [s[1] for s in S]
    wx, wy = (max(xs) - min(xs)) / 3, (max(ys) - min(ys)) / 3
    cell = [min(int((a - min(xs)) / wx), 2) * 3 + min(int((b - min(ys)) / wy), 2) for a, b in zip(xs, ys)]
    for i in range(N):
        for j in range(N):
            if cell[i] == cell[j]:
                assert folds[i] == folds[j]
    assert set(folds) <= {0, 1, 2}
    for f in set(folds):
        tr = [j for j in range(N) if folds[j] != f]
        b = _ols([Z[j] for j in tr], [Y[j] for j in tr])
        for i in range(N):
            if folds[i] == f:
                assert abs(r.extra["predictions"][i] - _pred(b, Z[i])) < 1e-9
