"""Tests for morie.fn.zxscv: leave-one-out recomputed by refitting."""

import math

from morie.fn.zxscv import spatial_cv_loo

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


def test_hat_matrix_equals_refitting():
    r = spatial_cv_loo(Y, X, S)
    for i in range(N):
        b = _ols([Z[j] for j in range(N) if j != i], [Y[j] for j in range(N) if j != i])
        assert abs(r.extra["predictions"][i] - _pred(b, Z[i])) < 1e-9
    assert abs(r.value - math.sqrt(sum((a - b) ** 2 for a, b in zip(Y, r.extra["predictions"])) / N)) < 1e-12
