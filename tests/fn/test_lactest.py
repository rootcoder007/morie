"""Tests for morie.fn.lactest: expected values recomputed from the formulas."""

import math

from morie.fn.lactest import lactest

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


def test_moran_randomisation_moments():
    n, S0, S1, S2 = _consts(W)
    m = sum(Y) / n
    z = [v - m for v in Y]
    zz = sum(v * v for v in z)
    moran_i = n / S0 * sum(a * b for a, b in zip(z, _lag(z))) / zz
    K = n * sum(v**4 for v in z) / zz**2
    E = -1 / (n - 1)
    V = (n * (S1 * (n * n - 3 * n + 3) - n * S2 + 3 * S0**2) - K * (S1 * (n * n - n) - 2 * n * S2 + 6 * S0**2)) / (
        (n - 1) * (n - 2) * (n - 3) * S0**2
    ) - E * E
    r = lactest(Y, W)
    assert abs(r.statistic - moran_i) < 1e-13
    assert abs(r.variance - V) < 1e-13
    assert abs(r.p_value - _upper((moran_i - E) / math.sqrt(V))) < 1e-12
    assert r.extra["geary"]["expected"] == 1.0
    assert r.extra["getis_ord"] is not None
    assert lactest([v - 2 for v in Y], W).extra["getis_ord"] is None
