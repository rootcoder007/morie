"""Tests for morie.fn.swpath."""

import math

from morie.fn.swpath import swpath

# ring of 6 plus a chord 0-3 with length 2.5
W = [[0.0] * 7 for _ in range(7)]
for a in range(6):
    b = (a + 1) % 6
    W[a][b] = W[b][a] = 1.0
W[0][3] = W[3][0] = 2.5


def test_orders():
    assert swpath(W, 0, 3) == 1
    assert swpath(W, 1, 4) == 3
    assert swpath(W, 1, 3) == 2
    assert swpath(W, 0, 6) == math.inf


def test_dijkstra():
    assert swpath(W, 0, 3, weighted=True) == 2.5
    assert swpath(W, 1, 4, weighted=True) == 3.0
