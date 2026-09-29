"""Tests for morie.fn.lacbivl: expected values recomputed from the formulas."""

import math

from morie.fn.lacbivl import lacbivl

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


def test_lee_l_and_local_components():
    mx, my = sum(X2) / N, sum(Y) / N
    zx, zy = [a - mx for a in X2], [a - my for a in Y]
    lx, ly = _lag(zx), _lag(zy)
    s = math.sqrt(sum(v * v for v in zx)) * math.sqrt(sum(v * v for v in zy))
    L = N / sum(sum(row) ** 2 for row in W) * sum(a * b for a, b in zip(lx, ly)) / s
    r = lacbivl(X2, Y, W)
    assert abs(r.statistic - L) < 1e-13
    assert max(abs(a - N * b * c / s) for a, b, c in zip(r.local_values, lx, ly)) < 1e-13
