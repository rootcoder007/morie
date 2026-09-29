"""Tests for morie.fn.sfgetis: recompute the Getis filter."""

from morie.fn.sfgetis import sfgetis

N = 6
W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(N)] for i in range(N)]
Y = [2.0, 4.0, 6.0, 3.0, 5.0, 1.5]


def test_filter():
    r = sfgetis(Y, W)
    tot = sum(Y)
    for i in range(N):
        wi = sum(W[i][j] for j in range(N) if j != i)
        g = sum(W[i][j] * Y[j] for j in range(N) if j != i) / (tot - Y[i])
        ref = Y[i] * (wi / (N - 1)) / g
        assert abs(r["filtered"][i] - ref) < 1e-12
        assert abs(r["spatial"][i] - (Y[i] - ref)) < 1e-12
