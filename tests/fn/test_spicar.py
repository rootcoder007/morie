"""Tests for spicar.schabenberger_icar_prior."""

import math

import pytest

NR, NC = 5, 4
N = NR * NC
# rook adjacency on a 5 x 4 grid, sites numbered row by row
W = [[1.0 if abs(i // NC - j // NC) + abs(i % NC - j % NC) == 1 else 0.0
      for j in range(N)] for i in range(N)]
Z = [((i * 37) % 17) / 3 + 0.5 * (i // NC) for i in range(N)]
X1 = [float((i * 5) % 9) for i in range(N)]
X2 = [math.cos(i) for i in range(N)]
Y = [1 + 0.8 * a - 0.5 * b + ((i * 13) % 7 - 3) / 2 + 0.3 * (i // NC)
     for i, (a, b) in enumerate(zip(X1, X2))]
X = [[1.0, a, b] for a, b in zip(X1, X2)]

from morie.fn.spicar import schabenberger_icar_prior


def test_spicar_basic():
    """Q = (D - W)/tau2, rank n - 1 on a connected grid, improper."""
    r = schabenberger_icar_prior(W, tau2=2.0)
    q = [list(row) for row in r["Q"]]
    for i in range(N):
        for j in range(N):
            ref = (sum(W[i]) if i == j else 0.0) - W[i][j]
            assert q[i][j] == pytest.approx(ref / 2.0, abs=1e-15)
        assert abs(sum(q[i])) < 1e-15
    assert r["rank"] == N - 1
    assert r["n_components"] == 1
    assert r["is_improper"]


def test_spicar_edge():
    """Two disconnected blocks lose two ranks."""
    w = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    r = schabenberger_icar_prior(w)
    assert r["n_components"] == 2
    assert r["rank"] == 2
