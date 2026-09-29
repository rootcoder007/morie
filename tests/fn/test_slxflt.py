"""Tests for morie.fn.slxflt: every expected value is recomputed from the formula."""

from morie.fn.slxflt import slxflt

N = 8
_B = [[1.0 if abs(i - j) == 1 or {i, j} == {0, 7} or {i, j} == {2, 5} else 0.0 for j in range(N)] for i in range(N)]
W = [[v / sum(r) for v in r] for r in _B]
X = [[1.0, ((i * 7) % 11) / 5] for i in range(N)]
Y = [1 + 2 * X[i][1] + ((i * 3) % 5 - 2) / 4 + 0.3 * sum(W[i][j] * X[j][1] for j in range(N)) for i in range(N)]
E = [((i * 4) % 7 - 3) / 5 for i in range(N)]
W3 = [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]


def test_local_deviation():
    r = slxflt(X, W)
    assert r.statistic == 2.0
    for i in range(N):
        for k in range(2):
            want = X[i][k] - sum(W[i][j] * X[j][k] for j in range(N))
            assert abs(r.extra["filtered"][i][k] - want) < 1e-14
