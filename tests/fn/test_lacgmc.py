"""Tests for morie.fn.lacgmc: expected values recomputed from the formulas."""

import math

from morie.fn._rng import random_uniform
from morie.fn.lacgmc import lacgmc

N = 9
_B = [[1.0 if abs(i - j) == 1 or {i, j} == {0, 8} or {i, j} == {2, 6} else 0.0 for j in range(N)] for i in range(N)]
W = [[v / sum(r) for v in r] for r in _B]
Y = [1.0, 2.4, 1.3, 3.1, 1.9, 2.2, 0.7, 2.8, 1.6]
X2 = [0.5, 0.9, 0.2, 1.4, 0.8, 1.1, 0.3, 1.2, 0.6]


def _lag(v, Wm=W):
    return [sum(a * b for a, b in zip(r, v)) for r in Wm]


def _consts(Wm):
    n = len(Wm)
    S0 = sum(map(sum, Wm))
    S1 = 0.5 * sum((Wm[i][j] + Wm[j][i]) ** 2 for i in range(n) for j in range(n))
    S2 = sum((sum(Wm[i]) + sum(Wm[j][i] for j in range(n))) ** 2 for i in range(n))
    return n, S0, S1, S2


def _upper(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def _geary(v):
    n, S0, _, _ = _consts(W)
    m = sum(v) / n
    return (
        (n - 1)
        * sum(W[i][j] * (v[i] - v[j]) ** 2 for i in range(n) for j in range(n))
        / (2 * S0 * sum((t - m) ** 2 for t in v))
    )


def test_permutation_distribution():
    nsim = 29
    u = random_uniform(nsim * N, seed=3)
    u = u.tolist() if hasattr(u, "tolist") else list(u)
    sims = []
    for s in range(nsim):
        blk = u[s * N : (s + 1) * N]
        order = sorted(range(N), key=lambda k: blk[k])
        sims.append(_geary([Y[k] for k in order]))
    C = _geary(Y)
    r = lacgmc(Y, W, nsim=nsim, seed=3)
    assert abs(r.statistic - C) < 1e-13
    assert max(abs(a - b) for a, b in zip(r.extra["simulated"], sims)) < 1e-13
    assert r.p_value == (1 + sum(1 for v in sims if v <= C)) / (nsim + 1)
