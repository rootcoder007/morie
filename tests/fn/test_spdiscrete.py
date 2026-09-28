"""spdiscrete: score equations, moment conditions and the Moran variance from their definitions."""

import math

import pytest

from morie.fn._qpcore import solve
from morie.fn._rng import random_normal, random_uniform
from morie.fn._rrng_core import dnorm, pnorm
from morie.fn.spdiscrete import binary_glm, discrete_moran_test, spatial_logit_gmm, spatial_probit_gmm

N = 30
_u = [float(v) for v in random_uniform(2 * N, seed=9)]
W = [[0.0] * N for _ in range(N)]
for i in range(N):
    for j in sorted(range(N), key=lambda j: (_u[i] - _u[j]) ** 2 + (_u[N + i] - _u[N + j]) ** 2)[1:4]:
        W[i][j] = W[j][i] = 1.0
W = [[v / sum(r) for v in r] for r in W]
_z = [float(v) for v in random_normal(3 * N, seed=10)]
X = [[1.0, _z[i], _z[N + i]] for i in range(N)]
_A = [[(1.0 if i == j else 0.0) - 0.4 * W[i][j] for j in range(N)] for i in range(N)]
Y = [1.0 if v > 0 else 0.0 for v in solve(_A, [0.1 + X[i][1] - 0.6 * X[i][2] + _z[2 * N + i] for i in range(N)])]


def test_glm_score_equations():
    lg = binary_glm(Y, X, "logit")
    for a in range(3):
        assert abs(sum(X[i][a] * (Y[i] - lg.fitted[i]) for i in range(N))) < 1e-8
    pr = binary_glm(Y, X, "probit")
    for a in range(3):
        s = 0.0
        for i in range(N):
            e = sum(X[i][b] * pr.coefficients[b] for b in range(3))
            F, f = pnorm(e), dnorm(e)
            s += (Y[i] - F) * f / (F * (1 - F)) * X[i][a]
        assert abs(s) < 1e-6  # deviance-based stopping, as glm


def test_spatial_logit_is_ols_on_projected_gradients():
    r = spatial_logit_gmm(Y, X, W)
    b = binary_glm(Y, X, "logit")
    p, xb = b.fitted, b.linear_predictor
    g = [q * (1 - q) for q in p]
    Z = [X[i] + [sum(W[i][j] * X[j][c] for j in range(N)) for c in (1, 2)] for i in range(N)]
    zz = [[sum(r[a] * r[c] for r in Z) for c in range(5)] for a in range(5)]

    def proj(v):
        bb = solve(zz, [sum(Z[i][a] * v[i] for i in range(N)) for a in range(5)])
        return [sum(Z[i][a] * bb[a] for a in range(5)) for i in range(N)]

    wxb = [sum(W[i][j] * xb[j] for j in range(N)) for i in range(N)]
    cols = [[g[i] * X[i][a] for i in range(N)] for a in range(3)] + [[g[i] * wxb[i] for i in range(N)]]
    G = [list(r) for r in zip(*[proj(c) for c in cols])]
    u = [Y[i] - p[i] + sum(cols[a][i] * b.coefficients[a] for a in range(3)) for i in range(N)]
    coef = solve(
        [[sum(r[a] * r[c] for r in G) for c in range(4)] for a in range(4)],
        [sum(r[a] * v for r, v in zip(G, u)) for a in range(4)],
    )
    assert r.coefficients + [r.rho] == pytest.approx(coef, rel=1e-10)
    assert all(s > 0 for s in r.se)


def test_probit_gmm_converges_to_a_fixed_point():
    r = spatial_probit_gmm(Y, X, W)
    again = spatial_probit_gmm(Y, X, W, start_rho=r.rho, tol=1e-12)
    assert again.rho == pytest.approx(r.rho, abs=1e-8)
    assert -1 < r.rho < 1 and r.iterations < 500


def test_moran_variance_by_definition():
    for link in ("logit", "probit"):
        t = discrete_moran_test(Y, X, W, link)
        fit = binary_glm(Y, X, link)
        if link == "logit":
            u = [y - p for y, p in zip(Y, fit.fitted)]
            s2 = [p * (1 - p) for p in fit.fitted]
        else:
            f = [dnorm(e) for e in fit.linear_predictor]
            u = [(y - F) * d / (F * (1 - F)) for y, F, d in zip(Y, fit.fitted, f)]
            s2 = [d * d / (F * (1 - F)) for F, d in zip(fit.fitted, f)]
        var = sum((W[i][j] ** 2 + W[i][j] * W[j][i]) * s2[i] * s2[j] for i in range(N) for j in range(N) if i != j)
        num = sum(u[i] * W[i][j] * u[j] for i in range(N) for j in range(N))
        assert t.statistic == pytest.approx(num / math.sqrt(var), rel=1e-10)
