"""Tests for morie.fn.swlag."""

from morie.fn.swlag import swlag

W = [[0, 0.5, 0.5, 0], [1 / 3, 0, 1 / 3, 1 / 3], [0.5, 0.5, 0, 0], [0, 1, 0, 0]]
Y = [1.5, -2.0, 4.0, 0.25]


def test_lag():
    ref = [sum(W[i][j] * Y[j] for j in range(4)) for i in range(4)]
    assert all(abs(a - b) < 1e-15 for a, b in zip(swlag(W, Y), ref))
