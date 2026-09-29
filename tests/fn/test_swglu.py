"""Tests for swglu.swiglu: (x W1 * SiLU(x W3)) W2 recomputed."""

import math

from morie.fn.swglu import swiglu


def _silu(v):
    return v / (1 + math.exp(-v))


def test_projected():
    x = [[0.5, -1.0, 2.0]]
    W1 = [[0.1, 0.2], [0.3, -0.4], [0.5, 0.6]]
    W3 = [[-0.2, 0.7], [0.1, 0.1], [0.4, -0.3]]
    W2 = [[1.0, -1.0, 0.5], [0.25, 0.5, 2.0]]
    up = [sum(x[0][i] * W1[i][j] for i in range(3)) for j in range(2)]
    gate = [sum(x[0][i] * W3[i][j] for i in range(3)) for j in range(2)]
    h = [u * _silu(g) for u, g in zip(up, gate)]
    ref = [sum(h[j] * W2[j][k] for j in range(2)) for k in range(3)]
    out = swiglu(x, W1, W2, W3).value
    assert max(abs(float(out[0][k]) - ref[k]) for k in range(3)) < 1e-14


def test_elementwise():
    out = swiglu([[1.0, -2.0]]).value
    assert abs(float(out[0][0]) - _silu(1.0)) < 1e-15 and abs(float(out[0][1]) + 2 * _silu(-2.0)) < 1e-15
