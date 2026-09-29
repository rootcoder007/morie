"""Tests for morie.fn.lapkn: the exponential kernel recomputed entry by entry."""

import math

from morie.fn.lapkn import expkern, laplacian_kernel

X = [[0.1, 1.2, -0.3], [0.8, 0.4, 0.5], [-1.0, 0.0, 2.0], [0.3, -0.7, 1.1]]
Z = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]


def test_gram_matrix_default_gamma():
    r = laplacian_kernel(X)
    for i in range(4):
        for j in range(4):
            assert abs(r["K"][i][j] - math.exp(-math.dist(X[i], X[j]) / 3)) < 1e-15
    assert r["gamma"] == 1 / 3


def test_cross_kernel():
    r = expkern(X, gamma=0.7, Z=Z)
    assert (r["n"], r["m"]) == (4, 2)
    assert abs(r["K"][2][1] - math.exp(-0.7 * math.dist(X[2], Z[1]))) < 1e-15
