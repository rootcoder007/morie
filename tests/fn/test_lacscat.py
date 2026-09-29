"""Tests for morie.fn.lacscat: expected values recomputed from the formulas."""

import math

from morie.fn.lacscat import lacscat

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


def test_quadrants_and_slope():
    m = sum(Y) / N
    z = [v - m for v in Y]
    lz = _lag(z)
    q = [(1 if b > 0 else 4) if a > 0 else (2 if b > 0 else 3) for a, b in zip(z, lz)]
    r = lacscat(Y, W)
    assert r.local_values == q
    assert r.extra["counts"] == [q.count(k) for k in (1, 2, 3, 4)]
    assert abs(r.statistic - sum(a * b for a, b in zip(z, lz)) / sum(a * a for a in z)) < 1e-14
