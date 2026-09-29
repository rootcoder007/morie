"""Tests for morie.fn.slxwx: every expected value is recomputed from the formula."""

import math

from morie.fn.slxwx import slxwx

N = 8
_B = [[1.0 if abs(i - j) == 1 or {i, j} == {0, 7} or {i, j} == {2, 5} else 0.0 for j in range(N)] for i in range(N)]
W = [[v / sum(r) for v in r] for r in _B]
X = [[1.0, ((i * 7) % 11) / 5] for i in range(N)]
Y = [1 + 2 * X[i][1] + ((i * 3) % 5 - 2) / 4 + 0.3 * sum(W[i][j] * X[j][1] for j in range(N)) for i in range(N)]
E = [((i * 4) % 7 - 3) / 5 for i in range(N)]
W3 = [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]


def test_lagged_columns_means_and_correlations():
    r = slxwx(X, W)
    assert r.statistic == 1.0 and r.extra["lagged_columns"] == [1]
    wx = [sum(W[i][j] * X[j][1] for j in range(N)) for i in range(N)]
    assert max(abs(a[0] - b) for a, b in zip(r.extra["WX"], wx)) < 1e-14
    x = [row[1] for row in X]
    mx, mw = sum(x) / N, sum(wx) / N
    c = sum((a - mx) * (b - mw) for a, b in zip(x, wx)) / math.sqrt(
        sum((a - mx) ** 2 for a in x) * sum((b - mw) ** 2 for b in wx)
    )
    assert abs(r.extra["means"][0] - mw) < 1e-14
    assert abs(r.extra["correlations"][0] - c) < 1e-12
    assert r.extra["design"][3] == X[3] + [wx[3]]
