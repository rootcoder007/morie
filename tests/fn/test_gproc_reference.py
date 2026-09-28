"""gproc: GP formulas against direct linear algebra, LOO identity, Laplace fixed point, sparse limits."""

import math

import pytest

from morie.fn._rng import random_normal
from morie.fn.gproc import (
    deep_gp_sample,
    gp_classify,
    gp_covariance,
    gp_fit,
    gp_loo,
    gp_predict,
    gp_sample,
    gp_sparse,
    kumaraswamy_warp,
)

X = [(0.0,), (0.4,), (1.1,), (1.7,), (2.5,)]
Y = [0.2, 0.7, 0.9, 0.3, -0.4]


def test_kernels():
    K = gp_covariance([(0.0, 0.0)], [(1.0, 2.0)], "ard", lengthscale=[1.0, 2.0], variance=2.0)
    assert K[0][0] == pytest.approx(2 * math.exp(-0.5 * (1 + 1)))
    assert gp_covariance([(0.0,)], [(0.5,)], "periodic", period=1.0, lengthscale=1.0)[0][0] == pytest.approx(
        math.exp(-2)
    )
    assert gp_covariance([(1.0, 2.0)], [(3.0, -1.0)], "linear", variance=0.5, bias=1.0)[0][0] == pytest.approx(1.5)
    r = 2.0
    assert gp_covariance([(0.0,)], [(2.0,)], "matern52")[0][0] == pytest.approx(
        (1 + math.sqrt(5) * r + 5 * r * r / 3) * math.exp(-math.sqrt(5) * r)
    )
    assert gp_covariance([(0.0,)], [(2.0,)], "rq", alpha=2.0)[0][0] == pytest.approx((1 + 4 / 4) ** -2)
    with pytest.raises(ValueError):
        gp_covariance([(0.0,)], [(1.0,)], "bogus")


def solve(A, b):
    n = len(A)
    M = [row[:] + [v] for row, v in zip(A, b)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def test_regression_loo_and_evidence():
    r = gp_predict(X, Y, [(0.8,)], noise=0.05, lengthscale=0.7)
    K = [
        [math.exp(-0.5 * ((a[0] - b[0]) / 0.7) ** 2) + (0.05 if i == j else 0) for j, b in enumerate(X)]
        for i, a in enumerate(X)
    ]
    ks = [math.exp(-0.5 * ((a[0] - 0.8) / 0.7) ** 2) for a in X]
    al = solve(K, Y)
    assert r.mean[0] == pytest.approx(sum(k * a for k, a in zip(ks, al)), abs=1e-12)
    v = solve(K, ks)
    assert r.variance[0] == pytest.approx(1 - sum(k * a for k, a in zip(ks, v)), abs=1e-12)
    loo = gp_loo(X, Y, noise=0.05, lengthscale=0.7)
    for i in range(len(X)):
        Xo, Yo = X[:i] + X[i + 1 :], Y[:i] + Y[i + 1 :]
        p = gp_predict(Xo, Yo, [X[i]], noise=0.05, lengthscale=0.7)
        assert loo.mean[i] == pytest.approx(p.mean[0], abs=1e-10)
        assert loo.variance[i] == pytest.approx(p.variance[0] + 0.05, abs=1e-10)
    f = gp_fit(X, Y, kernels=("se", "matern52"))
    best = f.fits[f.best]
    th = best["params"]
    again = gp_predict(
        X, Y, [], kernel=f.best, noise=th["noise"], variance=th["variance"], lengthscale=th["lengthscale"]
    )
    assert again.log_marginal_likelihood == pytest.approx(best["log_marginal_likelihood"], abs=1e-9)
    for d in (0.01, -0.01):  # a local maximum in each coordinate
        bumped = gp_predict(
            X,
            Y,
            [],
            kernel=f.best,
            noise=th["noise"],
            variance=th["variance"] * math.exp(d),
            lengthscale=th["lengthscale"],
        )
        assert bumped.log_marginal_likelihood <= best["log_marginal_likelihood"] + 1e-6


def test_samples():
    Q = [(0.0,), (0.5,), (1.0,)]
    s = gp_sample(Q, nsim=2, seed=4, jitter=1e-12)
    K = gp_covariance(Q, Q)
    e = [float(v) for v in random_normal(3, seed=4, stream=1)]
    # Cholesky factor of K + 1e-12 I written out for 3 x 3
    L = [[0.0] * 3 for _ in range(3)]
    L[0][0] = math.sqrt(1 + 1e-12)
    L[1][0] = K[1][0] / L[0][0]
    L[1][1] = math.sqrt(1 + 1e-12 - L[1][0] ** 2)
    L[2][0] = K[2][0] / L[0][0]
    L[2][1] = (K[2][1] - L[2][0] * L[1][0]) / L[1][1]
    L[2][2] = math.sqrt(1 + 1e-12 - L[2][0] ** 2 - L[2][1] ** 2)
    want = [sum(L[i][k] * e[k] for k in range(3)) for i in range(3)]
    assert s.samples[1] == pytest.approx(want, abs=1e-10)
    post = gp_sample([(0.0,)], data=(X, Y), noise=1e-8, nsim=3)
    assert all(abs(p[0] - 0.2) < 1e-3 for p in post.samples)  # pinned at an observed input
    d = deep_gp_sample(Q, [("se", {}), ("matern32", {"lengthscale": 0.5})], seed=2)
    assert len(d.layers) == 2 and d.output == d.layers[1]
    assert kumaraswamy_warp([0.25], 0.5, 2.0) == pytest.approx([1 - (1 - 0.5) ** 2])


def test_classification_and_sparse():
    Xc = [(-2.0,), (-1.0,), (-0.3,), (0.4,), (1.2,), (2.1,)]
    yc = [-1, -1, -1, 1, 1, 1]
    c = gp_classify(Xc, yc, [(-1.5,), (1.5,)], variance=4.0)
    # Laplace fixed point: f = K grad log p(y | f)
    K = gp_covariance(Xc, Xc, variance=4.0)
    grad = [(yy + 1) / 2 - 1 / (1 + math.exp(-f)) for yy, f in zip(yc, c.f_hat)]
    assert c.f_hat == pytest.approx([sum(K[i][j] * grad[j] for j in range(6)) for i in range(6)], abs=1e-6)
    assert c.probability[0] < 0.5 < c.probability[1]
    full = gp_predict(X, Y, [(0.8,)], noise=0.1)
    sp = gp_sparse(X, Y, X, [(0.8,)], noise=0.1, method="dtc")
    assert sp.mean[0] == pytest.approx(full.mean[0], abs=1e-7) and sp.variance[0] == pytest.approx(
        full.variance[0], abs=1e-7
    )
    assert sp.dtc_log_evidence == pytest.approx(full.log_marginal_likelihood, abs=1e-6)
    few = gp_sparse(X, Y, [(0.5,), (2.0,)], [(0.8,)], noise=0.1)
    assert few.vfe_bound <= full.log_marginal_likelihood + 1e-9
