"""Tests for morie.fn.rbfkn: the Gaussian kernel recomputed entry by entry."""

import math

from morie.fn.rbfkn import rbf_kernel, rbfkern

X = [[0.1, 1.2, -0.3], [0.8, 0.4, 0.5], [-1.0, 0.0, 2.0], [0.3, -0.7, 1.1]]


def test_gram_matrix_default_gamma():
    r = rbf_kernel(X)
    for i in range(4):
        for j in range(4):
            assert abs(r["K"][i][j] - math.exp(-(math.dist(X[i], X[j]) ** 2) / 3)) < 1e-15


def test_given_gamma():
    r = rbfkern(X, gamma=0.25)
    assert abs(r["K"][0][3] - math.exp(-0.25 * math.dist(X[0], X[3]) ** 2)) < 1e-15
