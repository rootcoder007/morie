"""Tests for morie.fn.sdmolsi: every expected value is recomputed from the formula."""

import math

from morie.fn.sdmolsi import sdmolsi

N = 8
_B = [[1.0 if abs(i - j) == 1 or {i, j} == {0, 7} or {i, j} == {2, 5} else 0.0 for j in range(N)] for i in range(N)]
W = [[v / sum(r) for v in r] for r in _B]
X = [[1.0, ((i * 7) % 11) / 5] for i in range(N)]
Y = [1 + 2 * X[i][1] + ((i * 3) % 5 - 2) / 4 + 0.3 * sum(W[i][j] * X[j][1] for j in range(N)) for i in range(N)]
E = [((i * 4) % 7 - 3) / 5 for i in range(N)]
W3 = [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]


def test_simple_regression_closed_form():
    x = [row[1] for row in X]
    mx, my = sum(x) / N, sum(Y) / N
    b1 = sum((a - mx) * (b - my) for a, b in zip(x, Y)) / sum((a - mx) ** 2 for a in x)
    b0 = my - b1 * mx
    e = [b - b0 - b1 * a for a, b in zip(x, Y)]
    s2 = sum(v * v for v in e) / N
    ll = -N / 2 * (math.log(2 * math.pi * s2) + 1)
    r = sdmolsi(Y, X)
    assert abs(r.extra["beta"][0] - b0) < 1e-12 and abs(r.extra["beta"][1] - b1) < 1e-12
    assert abs(r.extra["sigma2"] - s2) < 1e-13
    assert abs(r.statistic - ll) < 1e-12
    assert abs(r.extra["aic"] - (-2 * ll + 6)) < 1e-11
