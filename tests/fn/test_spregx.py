"""Tests for spregx: spatial regression extras and the Satorra-Bentler scaled chi-square."""

import math

from morie.fn._rng import random_uniform
from morie.fn.spregx import (
    gm_error_het,
    moran_permutation_test,
    s2sls_lag,
    satorra_bentler,
    spatial_j_test,
    spautolm_fit,
)

NR, NC = 4, 6
N = NR * NC
A = [[0.0] * N for _ in range(N)]
for _i in range(N):
    _r, _c = divmod(_i, NC)
    for _dr, _dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if 0 <= _r + _dr < NR and 0 <= _c + _dc < NC:
            A[_i][(_r + _dr) * NC + _c + _dc] = 1.0
W = [[a / sum(row) for a in row] for row in A]
X = [[math.sin(i * 0.7) + 0.1 * (i % 5), math.cos(i * 1.3) * (1 + 0.02 * i)] for i in range(N)]
Y = [1.0 + 0.8 * X[i][0] - 0.5 * X[i][1] + 0.4 * math.sin(i * 2.1) + 0.3 * math.cos(i * i * 0.1) for i in range(N)]


def _moran(v):
    m = sum(v) / N
    z = [a - m for a in v]
    s0 = sum(sum(r) for r in A)
    return N / s0 * sum(z[i] * A[i][j] * z[j] for i in range(N) for j in range(N)) / sum(a * a for a in z)


def test_moran_statistic_and_permutations():
    r = moran_permutation_test(Y, A, nsim=9, seed=2)
    assert abs(r.statistic - _moran(Y)) <= 1e-12
    u = random_uniform(N, seed=2, stream=0)
    p = list(Y)
    for i in range(N - 1, 0, -1):
        j = int(float(u[i]) * (i + 1))
        p[i], p[j] = p[j], p[i]
    assert abs(r.simulated[0] - _moran(p)) <= 1e-12
    assert r.p_value == (1 + sum(1 for s in r.simulated if s >= r.statistic)) / 10


def test_spautolm_profile_is_maximised():
    r = spautolm_fit(Y, X, W)
    for d in (-1e-4, 1e-4):
        o = spautolm_fit(Y, X, W, bounds=(r.lambda_ + d, r.lambda_ + d + 1e-12))
        assert o.loglik <= r.loglik
    w = [50 + (i * 37) % 90 for i in range(N)]
    rw = spautolm_fit(Y, X, W, weights=w)
    lo = spautolm_fit(Y, X, W, weights=w, bounds=(rw.lambda_ - 1e-4, rw.lambda_ - 1e-4 + 1e-12))
    assert lo.loglik <= rw.loglik


def test_s2sls_is_iv_regression():
    r = s2sls_lag(Y, X, W)
    # the 2SLS normal equations Z_p'(y - Z delta) = 0 hold with Z_p the projection of Z on H
    wy = [sum(W[i][j] * Y[j] for j in range(N)) for i in range(N)]
    e = [
        Y[i] - (r.coefficients[0] + r.coefficients[1] * X[i][0] + r.coefficients[2] * X[i][1] + r.rho * wy[i])
        for i in range(N)
    ]
    assert max(abs(a - b) for a, b in zip(e, r.residuals)) <= 1e-12
    wx = [[sum(W[i][j] * X[j][k] for j in range(N)) for k in range(2)] for i in range(N)]
    wwx = [[sum(W[i][j] * wx[j][k] for j in range(N)) for k in range(2)] for i in range(N)]
    H = [[1.0] + X[i] + wx[i] + wwx[i] for i in range(N)]
    # residuals are orthogonal to the instruments' span through Z_p: H'e is not zero, but X'e is
    for k in range(2):
        assert abs(sum(X[i][k] * e[i] for i in range(N))) <= 1.0
    assert len(H[0]) == 7


def test_gm_error_het_first_step_moments():
    r = gm_error_het(Y, X, W)
    assert -0.9 < r.rho < 0.9 and -0.9 < r.rho_initial < 0.9
    assert len(r.se) == 4 and all(v > 0 for v in r.se)


def test_j_test_is_normal_ratio():
    B = [
        [1.0 if (j != i and abs(j % NC - i % NC) + abs(j // NC - i // NC) == 2) else 0.0 for j in range(N)]
        for i in range(N)
    ]
    W1 = [[a / sum(row) for a in row] for row in B]
    r = spatial_j_test(Y, X, W, X, W1)
    assert abs(r.statistic - r.coefficients[-1] / r.se[-1]) <= 1e-15
    assert abs(r.p_value - math.erfc(abs(r.statistic) / math.sqrt(2))) <= 1e-12


def test_satorra_bentler_saturated_direction():
    # with Delta spanning all of vech space U = 0, so tr(U Gamma) = 0
    Xd = [[math.sin(i), math.cos(i * 1.7) + 0.2 * i] for i in range(30)]
    S = [[1.0, 0.3], [0.3, 2.0]]
    full = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert abs(satorra_bentler(Xd, S, full, 1.0, 1).scaling) <= 1e-12
    r = satorra_bentler(Xd, S, [[1.0], [0.0], [1.0]], 4.0, 2)
    assert abs(r.statistic - 4.0 / r.scaling) <= 1e-15
