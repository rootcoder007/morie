import math

from morie.fn._rng import random_uniform
from morie.fn._s03core import jacobi
from morie.fn.spcount import (
    _lagged_design,
    bym_variance_fraction,
    count_model_ic,
    sar_negbin,
    sar_poisson,
    sar_poisson_lm_test,
    sar_zip,
)


def _data():
    n_side, n = 5, 25
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        r, c = divmod(i, n_side)
        nb = [
            (r + a) * n_side + c + b for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= r + a < 5 and 0 <= c + b < 5
        ]
        for j in nb:
            W[i][j] = 1.0 / len(nb)
    u = [float(v) for v in random_uniform(3 * n, seed=21)]
    X = [[1.0, 2 * u[i] - 1] for i in range(n)]
    y = []
    for i in range(n):
        mu, k = math.exp(0.8 + 0.6 * X[i][1]), 0
        p = cdf = math.exp(-mu)
        while u[n + i] > cdf:
            k += 1
            p *= mu / k
            cdf += p
        y.append(0 if u[2 * n + i] < 0.2 else k)
    return y, X, W


def _pois_ll(y, X, W, rho, b):
    Xt = _lagged_design(X, W, rho)
    mu = [math.exp(sum(r[a] * b[a] for a in range(2))) for r in Xt]
    return sum(yi * math.log(m) - m - math.lgamma(yi + 1) for yi, m in zip(y, mu)), Xt, mu


def test_poisson_score_and_profile():
    y, X, W = _data()
    r = sar_poisson(y, X, W)
    ll, Xt, mu = _pois_ll(y, X, W, r.rho, r.coefficients)
    assert abs(ll - r.loglik) < 1e-9
    for a in range(2):
        assert abs(sum(Xt[i][a] * (y[i] - mu[i]) for i in range(25))) < 1e-8
    for d in (-1e-3, 1e-3):
        if not -0.99 <= r.rho + d <= 0.99:
            continue
        assert sar_poisson(y, X, W, rho_bounds=(r.rho + d, r.rho + d)).loglik <= r.loglik + 1e-12


def test_negbin_score_equations():
    y, X, W = _data()
    r = sar_negbin(y, X, W, rho_bounds=(0.2, 0.2))
    Xt = _lagged_design(X, W, 0.2)
    th, mu = r.theta, r.fitted
    for a in range(2):
        assert abs(sum(Xt[i][a] * (y[i] - mu[i]) / (1 + mu[i] / th) for i in range(25))) < 1e-7


def test_zip_is_a_local_maximum():
    y, X, W = _data()
    r = sar_zip(y, X, W, rho_bounds=(0.3, 0.3))
    Xt = _lagged_design(X, W, 0.3)

    def ll(b0, b1, g):
        pi = 1 / (1 + math.exp(-g))
        s = 0.0
        for i in range(25):
            mu = math.exp(b0 * Xt[i][0] + b1 * Xt[i][1])
            s += (
                math.log(pi + (1 - pi) * math.exp(-mu))
                if y[i] == 0
                else math.log(1 - pi) + y[i] * math.log(mu) - mu - math.lgamma(y[i] + 1)
            )
        return s

    b0, b1 = r.count_coefficients
    g = r.zero_coefficients[0]
    base = ll(b0, b1, g)
    assert abs(base - r.loglik) < 1e-9
    for d in ((1e-4, 0, 0), (-1e-4, 0, 0), (0, 1e-4, 0), (0, -1e-4, 0), (0, 0, 1e-4), (0, 0, -1e-4)):
        assert ll(b0 + d[0], b1 + d[1], g + d[2]) <= base + 1e-10


def test_lm_score_is_the_derivative():
    y, X, W = _data()
    r = sar_poisson_lm_test(y, X, W)
    b = sar_poisson(y, X, W, rho_bounds=(0.0, 0.0)).coefficients
    h = 1e-6
    fd = (_pois_ll(y, X, W, h, b)[0] - _pois_ll(y, X, W, -h, b)[0]) / (2 * h)
    assert abs(fd - r.score) < 1e-5 * max(1, abs(r.score))
    assert abs(r.p_value - math.erfc(math.sqrt(r.statistic / 2))) < 1e-15


def test_ic_and_bym_scaling():
    r = count_model_ic(-61.5, 3, 36)
    assert r.AIC == 129.0 and abs(r.BIC - (123.0 + 3 * math.log(36))) < 1e-12
    Q = [[2.0, -1.0, -1.0, 0.0], [-1.0, 2.0, -1.0, 0.0], [-1.0, -1.0, 3.0, -1.0], [0.0, 0.0, -1.0, 1.0]]
    vals, vecs = jacobi(Q)
    diag = [sum(vecs[i][k] ** 2 / vals[k] for k in range(4) if vals[k] > 1e-9) for i in range(4)]
    s = math.exp(sum(math.log(v) for v in diag) / 4)
    b = bym_variance_fraction(1.0, 3.0, Q=Q)
    assert abs(b.scale - s) < 1e-12 and abs(b.fraction - s / (s + 3)) < 1e-12
