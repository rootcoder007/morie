"""Tests for morie.fn.spprml: exactness of the GHK likelihood when Sigma is diagonal, and the optimum."""

import math

from morie.fn.spprml import _ghk_loglik, spprml

N = 10
W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(N)] for i in range(N)]
W = [[v / sum(r) for v in r] for r in W]
X = [[1.0, v] for v in (2.0, -1.0, 0.1, 1.5, 0.6, -0.4, 0.9, -1.3, 0.2, 1.1)]
Y = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1]


def test_ghk_exact_for_independent_latents():
    # rho = 0: the orthant probability is prod Phi(s_i x_i b) whatever the draws
    b = [-0.3, 0.8]
    U = [[(0.37 * (r + 1) * (i + 1)) % 1 for i in range(N)] for r in range(4)]
    ref = sum(
        math.log(0.5 * math.erfc(-(2 * y - 1) * (x[0] * b[0] + x[1] * b[1]) / math.sqrt(2))) for x, y in zip(X, Y)
    )
    assert abs(_ghk_loglik(b + [0.0], Y, X, W, U) - ref) < 1e-12


def test_simulated_optimum():
    from morie.fn._rng import random_uniform

    r = spprml(Y, X, W, nsim=10, seed=3)
    U = [[float(v) for v in random_uniform(N, seed=3, stream=k)] for k in range(10)]
    par = list(r["coefficients"]) + [math.atanh(r["rho"])]
    ll = _ghk_loglik(par, Y, X, W, U)
    assert abs(ll - r["loglik"]) < 1e-12
    for j in range(3):
        for d in (1e-3, -1e-3):
            q = list(par)
            q[j] += d
            assert _ghk_loglik(q, Y, X, W, U) <= ll + 1e-12
