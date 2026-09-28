"""Tests for starima: Pfeifer-Deutsch STARMA space-time models."""

import math

from morie.fn._regression_core import _chi2_sf
from morie.fn.starima import st_acf, st_lag, st_pacf, st_portmanteau, star_fit, starma_fit, starma_forecast

W1 = [[0, 0.5, 0.5, 0], [0.5, 0, 0, 0.5], [0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0]]
W2 = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
W = [W1, W2]
X = [[(t * 7 + i * 3) % 5 + 0.1 * ((t * i) % 4) + 0.01 * t for i in range(4)] for t in range(12)]
T, N = 12, 4


def _lag(z, order):
    if order == 0:
        return list(z)
    M = W[order - 1]
    return [sum(M[i][j] * z[j] for j in range(N)) for i in range(N)]


def _center(x):
    m = sum(v for row in x for v in row) / (len(x) * len(x[0]))
    return [[v - m for v in row] for row in x]


def _gamma(z, lag, h, s):
    return sum(sum(a * b for a, b in zip(_lag(z[t], lag), _lag(z[t + s], h))) for t in range(T - s)) / (N * (T - s))


def test_st_lag_is_matrix_product():
    z = [1.0, 2.0, 4.0, 8.0]
    assert st_lag(z, W, 1) == [3.0, 4.5, 4.5, 3.0]
    assert st_lag(z, W, 2) == [8.0, 4.0, 2.0, 1.0]
    assert st_lag(z, W, 0) == z


def test_st_acf_matches_definition():
    r = st_acf(X, W, max_lag_t=3)
    z = _center(X)
    assert r.acf[0][0] == 1.0
    for s in range(4):
        for lag in range(3):
            want = _gamma(z, lag, 0, s) / math.sqrt(_gamma(z, lag, lag, 0) * _gamma(z, 0, 0, 0))
            assert abs(r.acf[s][lag] - want) <= 1e-12


def test_st_pacf_first_order_yule_walker():
    # with spatial order 0 only, the k = 1 system is phi gamma_00(0) = gamma_00(1)
    r = st_pacf(X, W, max_lag_t=2, max_lag_s=0)
    z = _center(X)
    assert abs(r.pacf[0][0] - _gamma(z, 0, 0, 1) / _gamma(z, 0, 0, 0)) <= 1e-12
    # k = 2: solve the 2 x 2 Yule-Walker system by Cramer's rule
    g0, g1, g2 = (_gamma(z, 0, 0, s) for s in range(3))
    det = g0 * g0 - g1 * g1
    phi22 = (g0 * g2 - g1 * g1) / det
    assert abs(r.pacf[1][0] - phi22) <= 1e-12


def test_star_fit_is_stacked_least_squares():
    r = star_fit(X, W, [[0, 1]])
    z = _center(X)
    rows = [[_lag(z[t - 1], 0)[i], _lag(z[t - 1], 1)[i]] for t in range(1, T) for i in range(N)]
    y = [z[t][i] for t in range(1, T) for i in range(N)]
    a = sum(r_[0] * r_[0] for r_ in rows)
    b = sum(r_[0] * r_[1] for r_ in rows)
    d = sum(r_[1] * r_[1] for r_ in rows)
    e = sum(r_[0] * v for r_, v in zip(rows, y))
    f = sum(r_[1] * v for r_, v in zip(rows, y))
    det = a * d - b * b
    beta = [(e * d - b * f) / det, (a * f - b * e) / det]
    assert abs(r.coefficients[0][2] - beta[0]) <= 1e-10 and abs(r.coefficients[1][2] - beta[1]) <= 1e-10
    resid = [v - beta[0] * r_[0] - beta[1] * r_[1] for r_, v in zip(rows, y)]
    rss = sum(v * v for v in resid)
    n = len(y)
    assert abs(r.rss - rss) <= 1e-9
    assert abs(r.residuals[0][0] - resid[0]) <= 1e-10
    assert abs(r.sigma2 - rss / (n - 2)) <= 1e-9
    assert abs(r.loglik - (-0.5 * n * (math.log(2 * math.pi * rss / n) + 1))) <= 1e-9
    assert abs(r.aic - (-2 * r.loglik + 4)) <= 1e-9


def test_starma_css_reaches_the_least_squares_optimum():
    ls = star_fit(X, W, [[0, 1]])
    m = starma_fit(X, W, [[0, 1]], [])
    # least squares is the exact minimiser of the conditional sum of squares
    assert m.rss <= ls.rss + 1e-9
    assert all(abs(a[2] - b[2]) <= 1e-6 for a, b in zip(m.phi, ls.coefficients))  # optimiser tolerance


def test_starma_residual_recursion_and_nested_fit():
    ar_only = starma_fit(X, W, [[0, 1]], [])
    m = starma_fit(X, W, [[0, 1]], [[0]])
    assert m.rss <= ar_only.rss + 1e-9  # the AR model is nested in the ARMA model
    z = _center(X)
    phi = {(k, lag): c for k, lag, c in m.phi}
    theta = {(j, lag): c for j, lag, c in m.theta}
    e = [[0.0] * N for _ in range(T)]
    for t in range(1, T):
        pred = [0.0] * N
        for (k, lag), c in phi.items():
            zl = _lag(z[t - k], lag)
            pred = [p + c * v for p, v in zip(pred, zl)]
        for (j, lag), c in theta.items():
            if t - j >= 1:
                el = _lag(e[t - j], lag)
                pred = [p - c * v for p, v in zip(pred, el)]
        e[t] = [z[t][i] - pred[i] for i in range(N)]
    for t in range(1, T):
        for i in range(N):
            assert abs(m.residuals[t - 1][i] - e[t][i]) <= 1e-12
    assert abs(m.rss - sum(v * v for t in range(1, T) for v in e[t])) <= 1e-9


def test_forecast_one_step_by_hand():
    m = starma_fit(X, W, [[0, 1]], [[0]])
    f = starma_forecast(m, X, W, 1)
    z = _center(X)
    mean = sum(v for row in X for v in row) / (T * N)
    pred = [0.0] * N
    for k, lag, c in m.phi:
        pred = [p + c * v for p, v in zip(pred, _lag(z[T - k], lag))]
    for j, lag, c in m.theta:
        pred = [p - c * v for p, v in zip(pred, _lag(m.residuals[T - 1 - j], lag))]
    for i in range(N):
        assert abs(f.forecast[0][i] - (pred[i] + mean)) <= 1e-12


def test_portmanteau_recomputed_from_the_residual_acf():
    m = starma_fit(X, W, [[0, 1]], [])
    q = st_portmanteau(m.residuals, W, max_lag_t=2, n_params=2)
    a = st_acf(m.residuals, W, max_lag_t=2, center=False)
    Q = N * len(m.residuals) * sum(a.acf[s][lag] ** 2 for s in (1, 2) for lag in range(3))
    assert abs(q.statistic - Q) <= 1e-12
    assert q.df == 2 * 3 - 2
    assert abs(q.p_value - _chi2_sf(Q, 4)) <= 1e-12
