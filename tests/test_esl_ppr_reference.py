"""Projection pursuit regression (ESL 11.1-11.4): Gauss-Newton fixed point, ridge spline, prediction, R parity."""

import math

from morie.fn.eslppr import _ridge_eval, esl_projection_pursuit
from morie.fn.linsys import _householder_ls


def xy_data():
    X = [[math.sin(1.3 * i), math.cos(0.7 * i + 1), math.sin(0.37 * i * i)] for i in range(60)]
    return X, [a * b for a, b, _ in X]


def test_single_index_and_gauss_newton_fixed_point():
    X = [[i / 5 - 2, ((7 * i) % 11) / 5 - 1] for i in range(21)]
    y = [(a + b) ** 2 for a, b in X]
    r = esl_projection_pursuit(X, y, M=1, penalty=0.01)
    w = r["omega"][0]
    # true direction (1, 1)/sqrt 2, up to the spline's smoothing bias (about 1e-3 at this penalty)
    assert abs(abs(w[0]) - 1 / math.sqrt(2)) < 5e-3 and abs(w[0] - w[1]) < 5e-3
    # one more Gauss-Newton step (11.4) from the returned direction barely moves it
    term = r["terms"][0]
    ybar = sum(y) / len(y)
    v = [sum(a * b for a, b in zip(x, w)) for x in X]
    gd = [_ridge_eval(term, t) for t in v]
    rows = [[d * xx for xx in x] for (_, d), x in zip(gd, X)]
    z = [d * vi + (yi - ybar) - g for (g, d), vi, yi in zip(gd, v, y)]
    new, _ = _householder_ls(rows, z)
    nrm = math.sqrt(sum(t * t for t in new))
    assert 1 - abs(sum(a * b / nrm for a, b in zip(new, w))) < 1e-6


def test_linear_index_is_exact():
    X = [[i / 5 - 2, ((7 * i) % 11) / 5 - 1, math.sin(i)] for i in range(21)]
    y = [3 * (a - 2 * c) + 1 for a, _, c in X]  # a spline fits a line with no penalty bias
    r = esl_projection_pursuit(X, y, M=1, penalty=0.5)
    w = r["omega"][0]
    s = math.sqrt(5)
    assert max(abs(abs(a) - abs(b)) for a, b in zip(w, [1 / s, 0, 2 / s])) < 1e-10 and w[0] * w[2] < 0
    assert r["rss"] < 1e-18


def test_ridge_function_is_the_natural_spline():
    X, y = xy_data()
    r = esl_projection_pursuit(X, y, M=2, penalty=0.01)
    tm = r["terms"][1]
    u = tm["knots"]
    for k in (0, 7, len(u) - 1):
        assert abs(_ridge_eval(tm, u[k])[0] - tm["values"][k]) < 1e-12
    for t in (u[3] + 1e-3, (u[10] + u[11]) / 2, u[0] - 0.5, u[-1] + 0.5):
        e = 1e-6
        fd = (_ridge_eval(tm, t + e)[0] - _ridge_eval(tm, t - e)[0]) / (2 * e)
        assert abs(fd - _ridge_eval(tm, t)[1]) < 1e-6
    lo = [_ridge_eval(tm, u[0] - s)[0] for s in (1, 2, 3)]
    assert abs((lo[0] - lo[1]) - (lo[1] - lo[2])) < 1e-12  # linear beyond the end knots


def test_fit_prediction_and_parity():
    X, y = xy_data()
    r = esl_projection_pursuit(X, y, M=2, penalty=0.01, newdata=X)
    tss = sum((v - sum(y) / 60) ** 2 for v in y)
    assert r["rss_path"][0] > r["rss_path"][1] and r["rss"] < tss / 20
    assert max(abs(a - b) for a, b in zip(r["predicted"], r["fitted"])) < 1e-10
    assert abs(r["rss"] - sum(e * e for e in r["residuals"])) < 1e-12
    # R arm, same algorithm
    assert abs(r["rss_path"][0] - 2.14943580527484324) < 1e-9
    assert abs(r["rss_path"][1] - 0.36087256471880574) < 1e-9
