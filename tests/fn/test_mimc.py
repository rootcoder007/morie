"""Tests for morie.fn.mimc: recompute mi and the pseudo p-value from the simulated values."""

from morie.fn.mimc import mimc

N = 10
W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(N)] for i in range(N)]
Y = [2.0, 3.5, 3.0, 5.0, 4.5, 6.0, 5.5, 8.0, 7.0, 9.5]


def _moran(v):
    n = len(v)
    m = sum(v) / n
    z = [a - m for a in v]
    s0 = sum(sum(r) for r in W)
    return n / s0 * sum(z[i] * W[i][j] * z[j] for i in range(n) for j in range(n)) / sum(a * a for a in z)


def test_permutation_pvalue():
    r = mimc(Y, W, nsim=49, seed=3)
    assert abs(r.statistic - _moran(Y)) < 1e-13
    sims = r.extra["simulated"]
    assert len(sims) == 49
    assert r.p_value == (1 + sum(1 for s in sims if s >= r.statistic)) / 50
    assert all(-3 < s < 3 for s in sims)
