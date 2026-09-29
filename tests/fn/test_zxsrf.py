"""Tests for morie.fn.zxsrf: forest structure: in-bag interpolation and out-of-bag averaging."""

import math

from morie.fn._rng import random_uniform
from morie.fn.zxsrf import spatial_rf

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


def test_single_full_tree_interpolates_its_bootstrap_sample():
    r = spatial_rf(Y, X, S, n_trees=1, mtry=3, min_leaf=1, seed=7)
    u = random_uniform(N, seed=7, stream=0)
    u = u.tolist() if hasattr(u, "tolist") else list(u)
    boot = {min(int(v * N), N - 1) for v in u}
    for i in boot:
        assert abs(r.extra["fitted"][i] - Y[i]) < 1e-12
    for i in range(N):
        if i in boot:
            assert r.extra["oob"][i] != r.extra["oob"][i]
        else:
            assert abs(r.extra["oob"][i] - r.extra["fitted"][i]) < 1e-12


def test_oob_rmse_and_reproducibility():
    a = spatial_rf(Y, X, S, n_trees=12, min_leaf=3, seed=2)
    b = spatial_rf(Y, X, S, n_trees=12, min_leaf=3, seed=2)
    assert a.extra["fitted"] == b.extra["fitted"]
    ok = [i for i, v in enumerate(a.extra["oob"]) if v == v]
    assert abs(a.value - math.sqrt(sum((Y[i] - a.extra["oob"][i]) ** 2 for i in ok) / len(ok))) < 1e-12
