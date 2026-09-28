"""LM tests, local Getis-Ord, SLX, impacts and PPML gravity.

Checked against R: lm.RStests and localG (spdep), lmSLX and impacts
(spatialreg) to 5e-15, glm(poisson) and sandwich HC0 to 1e-8; the cross
tests in tests/cross/test-morie_vs_spdep.R and test-morie_vs_spatialreg.R
repeat that in R.
"""

import math

import pytest

from morie.fn.gravpp import gravity_ppml
from morie.fn.lacgetl import lacgetl
from morie.fn.lmtests import lm_spatial_tests
from morie.fn.spslx import slx_regression, spatial_impacts

N = 8
Y = [1.0, 2.2, 1.4, 3.1, 0.9, 2.0, 2.6, 1.1]
X = [[1.0, v] for v in (0.1, 0.6, 0.2, 0.9, 0.3, 0.5, 0.8, 0.4)]


def _w():
    W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(N)] for i in range(N)]
    return [[v / sum(r) for v in r] for r in W]


def _ols_resid(y, X):
    p = len(X[0])
    A = [[sum(r[a] * r[b] for r in X) for b in range(p)] for a in range(p)]
    c = [sum(r[a] * t for r, t in zip(X, y)) for a in range(p)]
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    b = [(A[1][1] * c[0] - A[0][1] * c[1]) / det, (A[0][0] * c[1] - A[1][0] * c[0]) / det]
    return [t - r[0] * b[0] - r[1] * b[1] for r, t in zip(X, y)]


def test_lm_tests_error_statistic_by_hand_and_sarma_identity():
    W = _w()
    r = lm_spatial_tests(Y, X, W)
    u = _ols_resid(Y, X)
    s2 = sum(v * v for v in u) / N
    T = sum(W[i][j] ** 2 + W[i][j] * W[j][i] for i in range(N) for j in range(N))
    uWu = sum(u[i] * W[i][j] * u[j] for i in range(N) for j in range(N)) / s2
    assert r["RSerr"]["statistic"] == pytest.approx(uWu**2 / T, abs=1e-12)
    assert r["SARMA"]["statistic"] == pytest.approx(r["adjRSlag"]["statistic"] + r["RSerr"]["statistic"], abs=1e-12)
    assert round(r["RSerr"]["statistic"], 6) == 0.003087 and round(r["RSlag"]["statistic"], 6) == 1.298358


def test_local_getis_ord_star_documented_and_symmetric():
    W = [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]]
    z = lacgetl([1.0, 2.0, 4.0, 8.0], W).local_values
    assert [round(v, 6) for v in z] == [-1.453631, -1.585258, 1.025755, 1.453631]
    # a constant shift of y leaves Gi* unchanged
    assert lacgetl([11.0, 12.0, 14.0, 18.0], W).local_values == pytest.approx(z, abs=1e-12)


def test_slx_is_ols_on_x_and_wx():
    W = _w()
    r = slx_regression(Y, X, W)
    wx = [sum(W[i][j] * X[j][1] for j in range(N)) for i in range(N)]
    Z = [[1.0, X[i][1], wx[i]] for i in range(N)]
    ZtZ = [[sum(r_[a] * r_[b] for r_ in Z) for b in range(3)] for a in range(3)]
    Zty = [sum(r_[a] * t for r_, t in zip(Z, Y)) for a in range(3)]
    fitted_normal = [sum(ZtZ[a][b] * r.coefficients[b] for b in range(3)) for a in range(3)]
    assert fitted_normal == pytest.approx(Zty, abs=1e-10)
    imp = r.impacts[0]
    assert imp["total"] == pytest.approx(r.coefficients[1] + r.coefficients[2], abs=1e-15)
    assert [round(v, 6) for v in r.coefficients] == [1.287716, 2.116493, -0.951624]


def test_impacts_total_is_beta_over_one_minus_rho_for_row_standardised_w():
    W = _w()
    r = spatial_impacts(0.35, [2.0, -1.0], W)
    assert r.total == pytest.approx([2.0 / 0.65, -1.0 / 0.65], abs=1e-12)
    assert [round(r_, 6) for r_ in (spatial_impacts(0.4, [2.0], [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]).direct[0],)] == [
        2.253968
    ]


def test_ppml_solves_the_score_equations():
    F = [12.0, 0.0, 30.0, 7.0, 55.0, 3.0]
    mo, md, d = [5, 5, 9, 9, 20, 20], [9, 20, 5, 20, 5, 9], [1.0, 3.0, 1.0, 2.0, 3.0, 2.0]
    r = gravity_ppml(F, mo, md, d)
    Xg = [[1.0, math.log(mo[i]), math.log(md[i]), math.log(d[i])] for i in range(6)]
    score = [sum(Xg[i][a] * (F[i] - r.fitted[i]) for i in range(6)) for a in range(4)]
    assert max(abs(s) for s in score) < 1e-8
    assert [round(v, 6) for v in r.coefficients] == [8.8362, -0.610957, -2.50985, 0.884215]
