import math

from morie.fn._qpcore import ssum
from morie.fn._rng import random_normal, random_uniform
from morie.fn.sarreg import spatial_regression_ml
from morie.fn.sptests import bp_test, sdm_ml, spatial_bp_test


def _data(n=40, seed=61):
    u = [float(v) for v in random_uniform(4 * n, seed=seed, stream=0)]
    z = [float(v) for v in random_normal(2 * n, seed=seed, stream=1)]
    P = [(u[i], u[n + i]) for i in range(n)]
    W = []
    for i in range(n):
        d = sorted(range(n), key=lambda j: (math.dist(P[i], P[j]), j))[1:5]
        W.append([0.25 if j in d else 0.0 for j in range(n)])
    x = z[:n]
    e = [v * (0.5 + u[2 * n + i]) for i, v in enumerate(z[n:])]
    Wx = [ssum(a * b for a, b in zip(r, x)) for r in W]
    rhs = [1 + 2 * x[i] - 0.5 * Wx[i] + e[i] for i in range(n)]
    y = rhs[:]
    for _ in range(200):
        y = [rhs[i] + 0.4 * ssum(a * b for a, b in zip(W[i], y)) for i in range(n)]
    return y, [[1.0, v] for v in x], W


def test_sdm_is_lag_on_augmented_design_and_impacts_add_up():
    y, X, W = _data()
    r = sdm_ml(y, X, W)
    Wx = [ssum(a * b[1] for a, b in zip(row, X)) for row in W]
    f = spatial_regression_ml(y, [row + [w] for row, w in zip(X, Wx)], W, model="lag")
    assert abs(r.rho - f.extra["rho"]) < 1e-15 and r.beta + r.theta == [float(v) for v in f.value]
    assert abs(r.direct[0] + r.indirect[0] - r.total[0]) < 1e-12
    assert abs(r.total[0] - (r.beta[1] + r.theta[0]) / (1 - r.rho)) < 1e-10
    assert abs(r.spillover_index[0] - r.indirect[0] / r.total[0]) < 1e-15


def test_breusch_pagan_formulas():
    e = [1, -1, 2, -2, 3, -3]
    Z = [[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1]]
    r = bp_test(e, Z)
    s2 = 28 / 6
    w = [v * v - s2 for v in e]
    fitted = [1 - s2, 1 - s2, 4 - s2, 4 - s2, 9 - s2, 9 - s2]
    assert abs(r.statistic - 6 * ssum(v * v for v in fitted) / ssum(v * v for v in w)) < 1e-12
    o = bp_test(e, Z, studentize=False)
    assert abs(o.statistic - 0.5 * ssum((v / s2) ** 2 for v in fitted)) < 1e-12
    assert o.df == 2 and 0 < o.p_value < 1


def test_spatial_bp_residuals():
    y, X, W = _data()
    for model in ("lag", "error"):
        r = spatial_bp_test(y, X, W, model=model)
        f = spatial_regression_ml(y, X, W, model=model)
        b = [float(v) for v in f.value]
        Wy = [ssum(a * v for a, v in zip(row, y)) for row in W]
        if model == "lag":
            e = [y[i] - f.extra["rho"] * Wy[i] - b[0] - b[1] * X[i][1] for i in range(40)]
        else:
            lam = f.extra["lambda"]
            Wx = [ssum(a * v[1] for a, v in zip(row, X)) for row in W]
            e = [y[i] - lam * Wy[i] - b[0] * (1 - lam) - b[1] * (X[i][1] - lam * Wx[i]) for i in range(40)]
        assert max(abs(a - c) for a, c in zip(r.residuals, e)) < 1e-12
