"""Tests for morie.fn.lacgear: expected values recomputed from the formulas."""

import math

from morie.fn.lacgear import lacgear

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


def test_geary_c_and_randomisation_variance():
    n, S0, S1, S2 = _consts(W)
    m = sum(Y) / n
    z = [v - m for v in Y]
    zz = sum(v * v for v in z)
    C = (n - 1) * sum(W[i][j] * (Y[i] - Y[j]) ** 2 for i in range(n) for j in range(n)) / (2 * S0 * zz)
    K = n * sum(v**4 for v in z) / zz**2
    V = (
        (n - 1) * S1 * (n * n - 3 * n + 3 - K * (n - 1))
        - 0.25 * (n - 1) * S2 * (n * n + 3 * n - 6 - K * (n * n - n + 2))
        + S0 * S0 * (n * n - 3 - K * (n - 1) ** 2)
    ) / (n * (n - 2) * (n - 3) * S0 * S0)
    r = lacgear(Y, W)
    assert abs(r.statistic - C) < 1e-13
    assert abs(r.variance - V) < 1e-13
    assert r.expected == 1.0
    assert abs(r.p_value - _upper((1 - C) / math.sqrt(V))) < 1e-12


def test_normality_variance():
    n, S0, S1, S2 = _consts(W)
    V = ((2 * S1 + S2) * (n - 1) - 4 * S0 * S0) / (2 * (n + 1) * S0 * S0)
    assert abs(lacgear(Y, W, randomisation=False).variance - V) < 1e-14
