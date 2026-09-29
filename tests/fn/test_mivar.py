"""Tests for morie.fn.mivar: recompute the Cliff-Ord variances from W."""

from morie.fn.mivar import mivar

N = 9
W = [[1.0 if abs(i - j) == 1 or (i, j) in ((0, 8), (8, 0)) else 0.0 for j in range(N)] for i in range(N)]
X = [1.0, 4.0, 2.5, 7.0, 3.0, 3.5, 9.0, 0.5, 2.0]


def _consts():
    s0 = sum(sum(r) for r in W)
    s1 = 0.5 * sum((W[i][j] + W[j][i]) ** 2 for i in range(N) for j in range(N))
    s2 = sum((sum(W[i]) + sum(W[j][i] for j in range(N))) ** 2 for i in range(N))
    return s0, s1, s2


def test_normality_variance():
    s0, s1, s2 = _consts()
    n = N
    want = (n * n * s1 - n * s2 + 3 * s0 * s0) / (s0 * s0 * (n * n - 1)) - 1 / (n - 1) ** 2
    assert abs(mivar(n, s0, s1, s2).statistic - want) < 1e-14


def test_randomisation_variance():
    s0, s1, s2 = _consts()
    n = N
    m = sum(X) / n
    z = [v - m for v in X]
    b2 = n * sum(v**4 for v in z) / sum(v * v for v in z) ** 2
    num = n * ((n * n - 3 * n + 3) * s1 - n * s2 + 3 * s0 * s0) - b2 * ((n * n - n) * s1 - 2 * n * s2 + 6 * s0 * s0)
    want = num / ((n - 1) * (n - 2) * (n - 3) * s0 * s0) - 1 / (n - 1) ** 2
    r = mivar(n, s0, s1, s2, b2=b2)
    assert abs(r.statistic - want) < 1e-14
    assert r.extra["assumption"] == "randomisation"
