"""Tests for morie.fn.scpboot: parametric bootstrap recomputed with Philox inversion draws."""

import math

from morie.fn._rng import random_uniform
from morie.fn.scpboot import scpboot
from morie.fn.spcount import sar_poisson

W = [[0, 1, 0, 0, 0], [0.5, 0, 0.5, 0, 0], [0, 0.5, 0, 0.5, 0], [0, 0, 0.5, 0, 0.5], [0, 0, 0, 1, 0]]
X = [[1, 0.1], [1, 0.4], [1, 0.5], [1, 0.9], [1, 0.3]]
Y = [1, 3, 4, 7, 2]


def _qpois(mu, u):
    k, c = 0, math.exp(-mu)
    while c < u:
        k += 1
        c += math.exp(-mu + k * math.log(mu) - math.lgamma(k + 1))
    return k


def test_draws_and_interval():
    B = 3
    f = sar_poisson(Y, X, W)
    u = random_uniform(B * 5, seed=2)
    u = u.tolist() if hasattr(u, "tolist") else list(u)
    draws = [sar_poisson([_qpois(f.fitted[i], u[b * 5 + i]) for i in range(5)], X, W).rho for b in range(B)]
    r = scpboot(Y, X, W, B=B, seed=2, level=0.5)
    # sar_poisson stops its golden search near 1e-8; scpboot polishes the score to machine precision
    assert abs(r.statistic - f.rho) < 1e-6
    assert max(abs(a - b) for a, b in zip(r.extra["draws"], draws)) < 1e-6
    s = sorted(draws)
    assert abs(r.extra["ci_lower"] - (s[0] + 0.5 * (s[1] - s[0]))) < 1e-6
    assert abs(r.extra["ci_upper"] - (s[1] + 0.5 * (s[2] - s[1]))) < 1e-6
