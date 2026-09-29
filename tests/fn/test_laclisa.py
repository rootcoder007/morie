"""Tests for morie.fn.laclisa: expected values recomputed from the formulas."""

import math

from morie.fn.laclisa import laclisa

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


def _local():
    n = N
    m = sum(Y) / n
    z = [v - m for v in Y]
    m2 = sum(v * v for v in z) / n
    lz = _lag(z)
    Ii = [a / m2 * b for a, b in zip(z, lz)]
    Wi = [sum(r) for r in W]
    Wi2 = [sum(v * v for v in r) for r in W]
    E = [-(z[i] ** 2) * Wi[i] / ((n - 1) * m2) for i in range(n)]
    V = [
        (z[i] / m2) ** 2 * n / (n - 2) * (Wi2[i] - Wi[i] ** 2 / (n - 1)) * (m2 - z[i] ** 2 / (n - 1)) for i in range(n)
    ]
    P = [2 * _upper(abs((a - b) / math.sqrt(c))) for a, b, c in zip(Ii, E, V)]
    return z, lz, Ii, E, V, P


def test_local_moran_conditional_moments():
    z, lz, Ii, E, V, P = _local()
    r = laclisa(Y, W)
    for i in range(N):
        assert abs(r.local_values[i] - Ii[i]) < 1e-13
        assert abs(r.extra["expected"][i] - E[i]) < 1e-14
        assert abs(r.extra["variance"][i] - V[i]) < 1e-13
        assert abs(r.extra["p_value"][i] - P[i]) < 1e-12
    assert abs(r.statistic - sum(Ii) / N) < 1e-13
