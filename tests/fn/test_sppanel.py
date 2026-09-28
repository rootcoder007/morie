"""sppanel: first-order conditions and special cases of the spatial panel likelihoods."""

import math

import pytest

from morie.fn._qpcore import inverse
from morie.fn._rng import random_normal
from morie.fn.sppanel import information_criteria, spatial_panel_ml, spatial_panel_re_lag

N, T = 7, 4
W = [[0.0] * N for _ in range(N)]
for i in range(N):
    for j in (i - 1, i + 1, i + 2):
        if 0 <= j < N:
            W[i][j] = 1.0
W = [[v / sum(r) for v in r] for r in W]
Z = [float(v) for v in random_normal(4 * N * T, seed=21)]
X = [[Z[k], Z[N * T + k] + 0.2 * (k % N)] for k in range(N * T)]
Y = [0.5 + X[k][0] - 0.4 * X[k][1] + 0.6 * Z[2 * N * T + k % N] + 0.4 * Z[3 * N * T + k] for k in range(N * T)]


def _lag(v):
    return [sum(W[i][j] * v[t * N + j] for j in range(N)) for t in range(T) for i in range(N)]


def _logdet(r):
    A = [[(1.0 if i == j else 0.0) - r * W[i][j] for j in range(N)] for i in range(N)]
    M = [row[:] for row in A]
    ld = 0.0
    for c in range(N):
        p = max(range(c, N), key=lambda q: abs(M[q][c]))
        M[c], M[p] = M[p], M[c]
        ld += math.log(abs(M[c][c]))
        for q in range(c + 1, N):
            f = M[q][c] / M[c][c]
            for k in range(c, N):
                M[q][k] -= f * M[c][k]
    return ld


def test_lag_model_score_is_zero_and_loglik_consistent():
    r = spatial_panel_ml(Y, X, W, N, "lag", "individual")

    # the profile log-likelihood is maximal at rho: compare neighbours
    def prof(rho):
        return spatial_panel_ml(Y, X, W, N, "lag", "individual", interval=(rho - 1e-13, rho + 1e-13)).loglik

    assert prof(r.rho) >= prof(r.rho + 0.01) and prof(r.rho) >= prof(r.rho - 0.01)
    e = r.residuals
    s2 = sum(v * v for v in e) / (N * T)
    ll = T * _logdet(r.rho) - N * T / 2 * math.log(2 * math.pi * s2) - N * T / 2
    assert r.loglik == pytest.approx(ll, rel=1e-12)
    # score: sum over periods of residual cross W y (within) equals sigma2 T tr((I - rho W)^{-1} W)
    ai = inverse([[(1.0 if i == j else 0.0) - r.rho * W[i][j] for j in range(N)] for i in range(N)])
    tr = sum(ai[i][k] * W[k][i] for i in range(N) for k in range(N))
    unit = [sum(Y[t * N + i] for t in range(T)) / T for i in range(N)]
    yt = [Y[t * N + i] - unit[i] for t in range(T) for i in range(N)]
    assert sum(a * b for a, b in zip(e, _lag(yt))) == pytest.approx(s2 * T * tr, rel=1e-8)


def test_error_and_durbin_and_lee_yu():
    e = spatial_panel_ml(Y, X, W, N, "error", "twoways")
    assert -0.99 < e.rho < 0.99
    d = spatial_panel_ml(Y, X, W, N, "durbin", "individual")
    assert len(d.coefficients) == 4
    ly = spatial_panel_ml(Y, X, W, N, "lag", "individual", lee_yu=True)
    plain = spatial_panel_ml(Y, X, W, N, "lag", "individual")
    assert ly.sigma2 == pytest.approx(plain.sigma2 * T / (T - 1), rel=1e-14)
    pooled = spatial_panel_ml(Y, X, W, N, "lag", "pooled")
    assert len(pooled.coefficients) == 3


def test_re_lag_phi_is_breusch_fixed_point():
    r = spatial_panel_re_lag(Y, X, W, N)
    b = r.coefficients
    wy = _lag(Y)
    d = [Y[k] - r.rho * wy[k] - b[0] - b[1] * X[k][0] - b[2] * X[k][1] for k in range(N * T)]
    md = [sum(d[t * N + i] for t in range(T)) / T for i in range(N)]
    q = sum((d[t * N + i] - md[i]) ** 2 for t in range(T) for i in range(N))
    pq = T * sum(v * v for v in md)
    assert r.phi == pytest.approx(min(1.0, math.sqrt(q / ((T - 1) * pq))), rel=1e-8)


def test_information_criteria():
    r = information_criteria(-40.0, 3, 50)
    assert (r.aic, r.bic) == (86.0, pytest.approx(80.0 + 3 * math.log(50), rel=1e-15))
    assert r.aicc == pytest.approx(86.0 + 24 / 46, rel=1e-15)
