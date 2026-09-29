"""Tests for morie.fn.miorig: recompute Moran's mi and the normality moments."""

import math

from morie.fn.miorig import miorig

N = 10
W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(N)] for i in range(N)]
Y = [2.0, 3.5, 3.0, 5.0, 4.5, 6.0, 5.5, 8.0, 7.0, 9.5]


def test_statistic_and_moments():
    n = N
    m = sum(Y) / n
    z = [v - m for v in Y]
    s0 = sum(sum(r) for r in W)
    mi = n / s0 * sum(z[i] * W[i][j] * z[j] for i in range(n) for j in range(n)) / sum(v * v for v in z)
    s1 = 0.5 * sum((W[i][j] + W[j][i]) ** 2 for i in range(n) for j in range(n))
    s2 = sum((2 * sum(W[i])) ** 2 for i in range(n))
    v = (n * n * s1 - n * s2 + 3 * s0 * s0) / (s0 * s0 * (n * n - 1)) - 1 / (n - 1) ** 2
    zz = (mi + 1 / (n - 1)) / math.sqrt(v)
    r = miorig(Y, W)
    assert abs(r.statistic - mi) < 1e-13
    assert abs(r.variance - v) < 1e-14
    assert abs(r.p_value - 0.5 * math.erfc(zz / math.sqrt(2))) < 1e-12
