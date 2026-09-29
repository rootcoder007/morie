"""Tests for morie.fn.sarfilt: every expected value is recomputed from the formula."""

from morie.fn.sarfilt import sarfilt

N = 8
_B = [[1.0 if abs(i - j) == 1 or {i, j} == {0, 7} or {i, j} == {2, 5} else 0.0 for j in range(N)] for i in range(N)]
W = [[v / sum(r) for v in r] for r in _B]
X = [[1.0, ((i * 7) % 11) / 5] for i in range(N)]
Y = [1 + 2 * X[i][1] + ((i * 3) % 5 - 2) / 4 + 0.3 * sum(W[i][j] * X[j][1] for j in range(N)) for i in range(N)]
E = [((i * 4) % 7 - 3) / 5 for i in range(N)]
W3 = [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]


def test_filter_is_y_minus_parameter_times_lag():
    r = sarfilt(Y, W, 0.35)
    want = [Y[i] - 0.35 * sum(W[i][j] * Y[j] for j in range(N)) for i in range(N)]
    assert max(abs(a - b) for a, b in zip(r.local_values, want)) < 1e-12
    assert r.extra["filtered"] == r.local_values
    assert r.statistic == 0.35


def test_default_parameter():
    r = sarfilt([1.0, 2.0, 4.0], W3)
    assert max(abs(a - b) for a, b in zip(r.local_values, [1 - 0.3 * 2, 2 - 0.3 * 2.5, 4 - 0.3 * 2])) < 1e-14
