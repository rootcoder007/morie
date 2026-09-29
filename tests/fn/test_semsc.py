"""Tests for morie.fn.semsc: every expected value is recomputed from the formula."""

import math

from morie.fn.semsc import semsc

N = 8
_B = [[1.0 if abs(i - j) == 1 or {i, j} == {0, 7} or {i, j} == {2, 5} else 0.0 for j in range(N)] for i in range(N)]
W = [[v / sum(r) for v in r] for r in _B]
X = [[1.0, ((i * 7) % 11) / 5] for i in range(N)]
Y = [1 + 2 * X[i][1] + ((i * 3) % 5 - 2) / 4 + 0.3 * sum(W[i][j] * X[j][1] for j in range(N)) for i in range(N)]
E = [((i * 4) % 7 - 3) / 5 for i in range(N)]
W3 = [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]


def test_lm_error_formula():
    T = sum(W[i][j] * W[i][j] + W[i][j] * W[j][i] for i in range(N) for j in range(N))
    eWe = sum(E[i] * W[i][j] * E[j] for i in range(N) for j in range(N))
    s2 = sum(v * v for v in E) / N
    lm = (eWe / s2) ** 2 / T
    r = semsc(E, W)
    assert abs(r.statistic - lm) < 1e-12
    assert abs(r.p_value - math.erfc(math.sqrt(lm / 2))) < 1e-12
    assert r.extra["df"] == 1
