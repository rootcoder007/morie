"""Tests for morie.fn.swlag2."""

from morie.fn.swlag2 import swlag2

W = [[0, 0.5, 0.5, 0], [1 / 3, 0, 1 / 3, 1 / 3], [0.5, 0.5, 0, 0], [0, 1, 0, 0]]
Y = [1.5, -2.0, 4.0, 0.25]


def test_second_order_lag():
    W2 = [[sum(W[i][m] * W[m][j] for m in range(4)) for j in range(4)] for i in range(4)]
    ref = [sum(W2[i][j] * Y[j] for j in range(4)) for i in range(4)]
    assert all(abs(a - b) < 1e-14 for a, b in zip(swlag2(W, Y), ref))
