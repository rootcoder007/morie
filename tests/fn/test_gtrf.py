"""Tests for gtrf.graph_transformer."""

import math

import pytest

from morie.fn.gtrf import laplacian, sparse_attention


ADJ = {0: [1], 1: [0, 2], 2: [1]}


def test_gtrf_basic():
    """L = I - D^(-1/2) A D^(-1/2) on the path 0 - 1 - 2."""
    L = laplacian(ADJ, 3)
    s = 1 / math.sqrt(2)
    for got, want in zip(L, [[1.0, -s, 0.0], [-s, 1.0, -s], [0.0, -s, 1.0]]):
        assert got == pytest.approx(want, rel=1e-15, abs=1e-15)
    assert laplacian(ADJ, 3, normalized=False) == [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]]


def test_gtrf_edge():
    """Attention is restricted to neighbours: node 0 sees only node 1, so
    its output is exactly node 1's value vector; node 1 mixes 0 and 2 by
    a softmax of scaled dot products."""
    H = [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]]
    I2 = [[1.0, 0.0], [0.0, 1.0]]
    out = sparse_attention(H, ADJ, I2, I2, I2)["output"]
    assert out[0] == [0.0, 1.0]
    s0 = sum(H[1][a] * H[0][a] for a in range(2)) / math.sqrt(2)
    s2 = sum(H[1][a] * H[2][a] for a in range(2)) / math.sqrt(2)
    w0 = math.exp(s0) / (math.exp(s0) + math.exp(s2))
    assert out[1] == pytest.approx([w0 * 1.0 + (1 - w0) * 2.0, w0 * 0.0 + (1 - w0) * 1.0], rel=1e-14)


