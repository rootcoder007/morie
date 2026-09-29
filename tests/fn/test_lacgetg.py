"""Tests for morie.fn.lacgetg: expected values recomputed from the formulas."""

import math

from morie.fn.lacgetg import lacgetg

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


def test_global_g_and_moments():
    n, S0, S1, S2 = _consts(_B)
    num = sum(_B[i][j] * Y[i] * Y[j] for i in range(n) for j in range(n))
    den = sum(Y[i] * Y[j] for i in range(n) for j in range(n) if i != j)
    G = num / den
    EG = S0 / (n * (n - 1))
    s1, s2, s3, s4 = (sum(v**k for v in Y) for k in (1, 2, 3, 4))
    B0 = (n * n - 3 * n + 3) * S1 - n * S2 + 3 * S0**2
    B1 = -((n * n - n) * S1 - 2 * n * S2 + 6 * S0**2)
    B2 = -(2 * n * S1 - (n + 3) * S2 + 6 * S0**2)
    B3 = 4 * (n - 1) * S1 - 2 * (n + 1) * S2 + 8 * S0**2
    B4 = S1 - S2 + S0**2
    V = (B0 * s2**2 + B1 * s4 + B2 * s1**2 * s2 + B3 * s1 * s3 + B4 * s1**4) / (
        (s1**2 - s2) ** 2 * n * (n - 1) * (n - 2) * (n - 3)
    ) - EG**2
    r = lacgetg(Y, _B)
    assert abs(r.statistic - G) < 1e-14
    assert abs(r.expected - EG) < 1e-15
    assert abs(r.variance - V) < 1e-13
    assert abs(r.p_value - _upper((G - EG) / math.sqrt(V))) < 1e-12


def test_negative_values_rejected():
    import pytest

    with pytest.raises(ValueError):
        lacgetg([-1.0] + Y[1:], W)
