import math

from morie.fn._qpcore import ssum
from morie.fn._rng import random_uniform
from morie.fn.gwrbas import gwr_basic
from morie.fn.gwrext import gw_summary, gwr_collinearity, gwr_f_tests, gwr_predict

U = [float(v) for v in random_uniform(150, seed=13, stream=0)]
P = [(10 * U[i], 10 * U[30 + i]) for i in range(30)]
X = [[1.0, U[60 + i], U[90 + i]] for i in range(30)]
Y = [1 + (1 + 0.2 * p[0]) * x[1] - 2 * x[2] + 0.2 * U[120 + i] for i, (p, x) in enumerate(zip(P, X))]


def test_collinearity_global_bandwidth_is_global():
    r = gwr_collinearity(X, P, 1e6, kernel="gaussian")
    assert max(abs(v - r.vif[0][0]) for row in r.vif for v in row[:1]) < 1e-9
    assert all(c >= 1 for c in r.local_cn)
    assert all(0 <= v <= 1 + 1e-12 for row in r.vdp for v in row)
    assert all(math.isnan(row[0]) for row in r.corr)


def test_predict_at_data_equals_gwr_basic():
    fit = gwr_basic(Y, X, P, 4.0)
    r = gwr_predict(Y, X, P, X[:5], P[:5], 4.0)
    assert max(abs(a - b) for a, b in zip(r.prediction, fit.fitted[:5])) < 1e-10
    assert all(v > r.sigma2 for v in r.variance)


def test_f_tests_formulas():
    r = gwr_f_tests(Y, X, P, 4.0)
    assert abs(r.F4 - r.rss_gwr / r.rss_ols) < 1e-15
    assert abs(r.F1 - (r.rss_gwr / r.delta1) / (r.rss_ols / 27)) < 1e-12
    g = gwr_f_tests(Y, X, P, 4.0, method="gwmodel")
    assert g.delta2 == 0 and math.isinf(g.F1_df[0]) and abs(g.F1 - r.F1) < 1e-15
    assert len(r.F3) == 3 and all(0 <= p <= 1 for p in r.F3_p)


def test_gw_summary_global_matches_moments():
    Z = [[x[1], x[2], y] for x, y in zip(X, Y)]
    r = gw_summary(Z, P, 1e7, kernel="gaussian")
    col = [z[0] for z in Z]
    m = ssum(col) / 30
    assert abs(r.mean[0][0] - m) < 1e-9
    assert abs(r.var[0][0] - ssum((v - m) ** 2 for v in col) / 30) < 1e-9
    c2 = [z[2] for z in Z]
    m2 = ssum(c2) / 30
    cov = ssum((a - m) * (b - m2) for a, b in zip(col, c2)) / 29
    assert abs(r.cov[0][1] - cov) < 1e-8
