"""Tests for morie.fn.semboot: every expected value is recomputed from the formula."""

import math

from morie.fn._rng import random_uniform
from morie.fn.sarreg import spatial_regression_ml
from morie.fn.semboot import semboot


def _inv(A):
    n = len(A)
    M = [list(map(float, r)) + [1.0 if i == j else 0.0 for j in range(n)] for i, r in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(n):
            if r != c:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [r[n:] for r in M]


def _mm(A, B):
    return [[sum(a * b for a, b in zip(r, c)) for c in zip(*B)] for r in A]


def _mv(A, v):
    return [sum(a * b for a, b in zip(r, v)) for r in A]


def _t(A):
    return [list(c) for c in zip(*A)]


def _imw(W, a):
    return [[(1.0 if i == j else 0.0) - a * W[i][j] for j in range(len(W))] for i in range(len(W))]


def _det(A):
    # Laplace expansion (small matrices only)
    if len(A) == 1:
        return A[0][0]
    return sum((-1) ** j * A[0][j] * _det([r[:j] + r[j + 1 :] for r in A[1:]]) for j in range(len(A)))


N = 8
_B = [[1.0 if abs(i - j) == 1 or {i, j} == {0, 7} or {i, j} == {2, 5} else 0.0 for j in range(N)] for i in range(N)]
W = [[v / sum(r) for v in r] for r in _B]
X = [[1.0, ((i * 7) % 11) / 5] for i in range(N)]
Y = [1 + 2 * X[i][1] + ((i * 3) % 5 - 2) / 4 + 0.3 * sum(W[i][j] * X[j][1] for j in range(N)) for i in range(N)]
E = [((i * 4) % 7 - 3) / 5 for i in range(N)]
W3 = [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]


def _aslist(v):
    return v.tolist() if hasattr(v, "tolist") else list(v)


def _q7(x, q):
    s = sorted(x)
    h = (len(s) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def _boot(Z, model, which, B, seed):
    f = spatial_regression_ml(Y, Z, W, model=model)
    rho, lam = f.extra.get("rho", 0.0), f.extra.get("lambda", 0.0)
    Zb = _mv(Z, list(f.value))
    Ay = [a - rho * b for a, b in zip(Y, _mv(W, Y))]
    r0 = [a - b for a, b in zip(Ay, Zb)]
    e = [a - lam * b for a, b in zip(r0, _mv(W, r0))]
    m = sum(e) / N
    e = [v - m for v in e]
    Ai, Bi = _inv(_imw(W, rho)), _inv(_imw(W, lam))
    u = [float(v) for v in _aslist(random_uniform(B * N, seed=seed))]
    out = []
    for b in range(B):
        es = [e[int(u[b * N + i] * N)] for i in range(N)]
        ys = _mv(Ai, [a + c for a, c in zip(Zb, _mv(Bi, es))])
        g = spatial_regression_ml(ys, Z, W, model=model)
        out.append(g.extra[which])
    return f.extra[which], out


def test_percentile_interval_of_the_residual_bootstrap():
    Z = X
    est, draws = _boot(Z, "error", "lambda", 6, 11)
    r = semboot(Y, X, W, B=6, seed=11, level=0.9)
    # Brent in spatial_regression_ml stops at ~1e-8 on a flat likelihood; the front end polishes the score
    assert abs(r.statistic - est) < 1e-6
    for a, b in zip(sorted(r.extra["draws"]), sorted(draws)):
        assert abs(a - b) < 1e-6
    d = r.extra["draws"]
    assert r.extra["ci_lower"] == _q7(d, 0.05) and r.extra["ci_upper"] == _q7(d, 0.95)
    m = sum(d) / len(d)
    assert abs(r.extra["se_boot"] - math.sqrt(sum((v - m) ** 2 for v in d) / (len(d) - 1))) < 1e-14
