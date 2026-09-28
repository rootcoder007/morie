"""Tests for morie.fn.spdurbin (SDEM, GNS, CAR, LR/Wald tests, residual Moran, log-Jacobian)."""

import math

import pytest

from morie.fn.sarreg import spatial_regression_ml
from morie.fn.spdurbin import (
    car_ml,
    gns_ml,
    log_jacobian,
    residual_moran,
    sdem_ml,
    spatial_lr_test,
    spatial_wald_test,
)
from morie.fn.spslx import spatial_impacts

n = 16
B = [[1.0 if abs(i // 4 - j // 4) + abs(i % 4 - j % 4) == 1 else 0.0 for j in range(n)] for i in range(n)]
W = [[v / sum(r) for v in r] for r in B]
X = [[1.0, (i * 7 % 11) / 5, ((i * 5) % 7) / 3 - 1] for i in range(n)]
y = [
    1 + 2 * X[i][1] - X[i][2] + 0.3 * sum(W[i][j] * X[j][1] for j in range(n)) + ((i * 3) % 5 - 2) / 4 for i in range(n)
]


def test_sdem_is_error_model_on_durbin_design():
    r = sdem_ml(y, X, W)
    from morie.fn.sarreg import _mv  # same summation as the module (builtin sum is compensated on py>=3.12)

    WX = [_mv(W, [r[k] for r in X]) for k in (1, 2)]
    Z = [X[i] + [WX[0][i], WX[1][i]] for i in range(n)]
    e = spatial_regression_ml(y, Z, W, model="error")
    assert r.loglik == pytest.approx(e.extra["loglik"], abs=1e-12)
    assert r.coefficients == pytest.approx(list(e.value), abs=1e-12)
    b = r.coefficients
    assert r.impacts["total"] == pytest.approx([b[1] + b[3], b[2] + b[4]], abs=1e-15)
    V = r.cov
    assert r.impacts["se_total"][0] == pytest.approx(math.sqrt(V[1][1] + V[3][3] + 2 * V[1][3]), abs=1e-15)


def test_gns_impacts_use_fitted_rho():
    r = gns_ml(y, X, W)
    b = r.coefficients
    imp = spatial_impacts(r.rho, b[1:3], W, b[3:5])
    assert r.impacts["direct"] == pytest.approx(list(imp["direct"]), abs=1e-15)


def test_car_lr_against_ols():
    r = car_ml(y, X, B)
    XtX = [[sum(a[p] * a[q] for a in X) for q in range(3)] for p in range(3)]
    Xty = [sum(a[p] * t for a, t in zip(X, y)) for p in range(3)]
    from morie.fn._qpcore import solve

    beta = solve(XtX, Xty)
    s2 = sum((t - sum(u * v for u, v in zip(a, beta))) ** 2 for a, t in zip(X, y)) / n
    ll0 = -n / 2 * (math.log(2 * math.pi * s2) + 1)
    assert r.lr_test["statistic"] == pytest.approx(2 * (r.loglik - ll0), abs=1e-9)
    assert r.lr_test["statistic"] >= 0


def test_tests_and_jacobian():
    lr = spatial_lr_test(-10.0, -12.5, 1)
    assert (lr.statistic, round(lr.pvalue, 6)) == (5.0, 0.025347)
    w = spatial_wald_test([0.3, 0.2], [[0.01, 0.0], [0.0, 0.01]])
    assert w.statistic == pytest.approx(13.0) and w.df == 2
    assert log_jacobian([[0, 1], [1, 0]], 0.5) == pytest.approx(math.log(0.75), abs=1e-15)
    assert log_jacobian([[0, 1], [1, 0]], 0.5, 0.5) == pytest.approx(2 * math.log(0.75), abs=1e-15)


def test_residual_moran_expectation():
    r = residual_moran(y, X, W)
    from morie.fn._qpcore import inverse

    k = 3
    XtXi = inverse([[sum(a[p] * a[q] for a in X) for q in range(k)] for p in range(k)])
    H = [[sum(X[i][p] * XtXi[p][q] * X[j][q] for p in range(k) for q in range(k)) for j in range(n)] for i in range(n)]
    trMW = sum(W[i][i] - sum(H[i][m] * W[m][i] for m in range(n)) for i in range(n))
    assert r.expected == pytest.approx(n / n * trMW / (n - k), abs=1e-14)
    assert r.variance > 0 and 0 <= r.pvalue <= 1
