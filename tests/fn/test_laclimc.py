"""Tests for morie.fn.laclimc: expected values recomputed from the formulas."""

import math

from morie.fn._rng import random_uniform
from morie.fn.laclimc import laclimc

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


def test_conditional_permutation_pvalues():
    nsim = 19
    m = sum(Y) / N
    z = [v - m for v in Y]
    m2 = sum(v * v for v in z) / N
    u = random_uniform(nsim * N * (N - 1), seed=2)
    u = u.tolist() if hasattr(u, "tolist") else list(u)
    r = laclimc(Y, W, nsim=nsim, seed=2)
    for i in range(N):
        Ii = z[i] / m2 * sum(W[i][j] * z[j] for j in range(N))
        oth = [j for j in range(N) if j != i]
        sims = []
        for s in range(nsim):
            blk = u[(s * N + i) * (N - 1) : (s * N + i + 1) * (N - 1)]
            order = sorted(range(N - 1), key=lambda k: blk[k])
            sims.append(z[i] / m2 * sum(W[i][oth[k]] * z[oth[order[k]]] for k in range(N - 1)))
        ge = sum(1 for v in sims if v >= Ii)
        le = sum(1 for v in sims if v <= Ii)
        assert r.extra["p_value"][i] == (1 + min(ge, le)) / (nsim + 1)
        assert abs(r.local_values[i] - Ii) < 1e-13
        assert abs(r.extra["mean_sim"][i] - sum(sims) / nsim) < 1e-13
