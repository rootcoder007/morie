"""Tests for morie.fn.sglm: redirects to likfit / glmm_laplace_fit and recomputes the GLS standard errors."""

import math

from morie.fn.sglm import spatial_glm
from morie.fn.sglmm import glmm_laplace_fit
from morie.fn.vgmods import likfit

P = [(float(i % 4), float(i // 4)) for i in range(12)]
X = [[1.0, p[0]] for p in P]
Y = [1.1, 1.9, 3.2, 3.8, 1.3, 2.2, 2.9, 4.1, 0.8, 2.1, 3.0, 4.2]


def _inv(A):
    n = len(A)
    M = [list(r) + [1.0 if i == j else 0.0 for j in range(n)] for i, r in enumerate(A)]
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


def test_gaussian_is_likfit_with_gls_standard_errors():
    dmax = max(math.dist(a, b) for a in P for b in P)
    f = likfit(Y, P, {"model": "Exp", "range": dmax / 3}, X=X)
    r = spatial_glm(X, Y, P)
    assert r["estimate"] == list(f["beta"])
    V = [
        [
            f["psill"] * math.exp(-math.dist(a, b) / f["range"]) + (f["nugget"] if i == j else 0.0)
            for j, b in enumerate(P)
        ]
        for i, a in enumerate(P)
    ]
    Vi = _inv(V)
    XtViX = [
        [sum(X[i][a] * Vi[i][j] * X[j][b] for i in range(12) for j in range(12)) for b in range(2)] for a in range(2)
    ]
    C = _inv(XtViX)
    assert abs(r["se"][1] - math.sqrt(C[1][1])) < 1e-10


def test_poisson_is_the_laplace_glmm():
    y = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8]
    g = glmm_laplace_fit(y, X, family="poisson", coords=P)
    r = spatial_glm(X, y, P, family="poisson")
    assert r["estimate"] == list(g["beta"])
    assert r["loglik"] == g["loglik"]
