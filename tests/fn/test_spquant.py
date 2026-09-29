"""spquant: LP regression quantiles are optimal and the two stages compose."""

import pytest

from morie.fn._rng import random_normal
from morie.fn.spquant import quantile_regression_lp, spatial_quantile_iv


def _obj(y, X, b, tau):
    return sum(
        tau * e if e >= 0 else (tau - 1) * e
        for e in (v - sum(r[a] * b[a] for a in range(len(b))) for r, v in zip(X, y))
    )


def test_regression_quantile_optimality():
    z = [float(v) for v in random_normal(90, seed=7)]
    X = [[1.0, z[i]] for i in range(30)]
    y = [0.5 + 2 * z[i] + z[30 + i] for i in range(30)]
    for tau in (0.2, 0.5, 0.9):
        r = quantile_regression_lp(y, X, tau)
        assert r.objective == pytest.approx(_obj(y, X, r.coefficients, tau), rel=1e-12)
        for d in ((0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01)):
            assert _obj(y, X, [r.coefficients[0] + d[0], r.coefficients[1] + d[1]], tau) >= r.objective - 1e-12
    med = quantile_regression_lp([3.0, 1.0, 7.0, 5.0, 4.0], [[1.0]] * 5)
    assert med.coefficients == pytest.approx([4.0])


def test_spatial_two_stage_composes():
    n = 20
    W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(n)] for i in range(n)]
    W = [[v / sum(r) for v in r] for r in W]
    z = [float(v) for v in random_normal(60, seed=8)]
    X = [[z[i]] for i in range(n)]
    y = [1 + z[i] + 0.5 * z[20 + i] for i in range(n)]
    r = spatial_quantile_iv(y, X, W, 0.5)
    wy = [sum(W[i][j] * y[j] for j in range(n)) for i in range(n)]
    Z = [[1.0, X[i][0], sum(W[i][j] * X[j][0] for j in range(n))] for i in range(n)]
    s1 = quantile_regression_lp(wy, Z, 0.5)
    assert r.first_stage == pytest.approx(s1.coefficients, abs=1e-12)
    s2 = quantile_regression_lp(y, [[1.0, X[i][0], wy[i] - s1.residuals[i]] for i in range(n)], 0.5)
    assert r.coefficients + [r.rho] == pytest.approx(s2.coefficients, abs=1e-12)
