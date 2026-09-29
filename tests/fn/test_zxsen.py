"""Tests for morie.fn.zxsen: stacking weights solve the non-negative least-squares problem."""

import math

from morie.fn.zxsen import spatial_stacking

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


def test_weights_are_nonnegative_and_fitted_is_the_weighted_sum():
    r = spatial_stacking(Y, X, S, n_blocks=2, k=2, seed=3, n_trees=6)
    w = r.extra["weights"]
    assert all(v >= 0 for v in w) and len(w) == 3
    oof = r.extra["oof"]
    res = [Y[i] - sum(a * b for a, b in zip(w, oof[i])) for i in range(N)]
    for j in range(3):
        g = sum(oof[i][j] * res[i] for i in range(N))
        if w[j] > 0:
            assert abs(g) < 1e-8
        else:
            assert g <= 1e-8
    assert abs(r.value - math.sqrt(sum(v * v for v in res) / N)) < 1e-12
