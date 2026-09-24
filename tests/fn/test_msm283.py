"""Verification tests for msm283.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 14, eq. 14.12 p.471, the rotated ridge regression. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm283 import penfreg


Y = [1.0, 2.0]
X = [[1.0, 0.0], [0.0, 1.0]]
P = [[1.0, 0.0], [0.0, 1.0]]


def test_an_unpenalised_fit_is_ordinary_least_squares():
    # with X the identity and mu the mean, beta is y - mu
    res = penfreg(Y, X, P, 0.0)
    assert res["mu"] == pytest.approx(1.5, rel=1e-12)
    assert list(res["beta"]) == pytest.approx([-0.5, 0.5], rel=1e-12)
    assert list(res["fitted"]) == pytest.approx(Y, rel=1e-12)
    assert res["sse"] == pytest.approx(0.0, abs=1e-12)


def test_the_original_coefficients_are_the_rotation_of_the_starred_ones():
    # beta = Gamma beta*
    res = penfreg(Y, X, P, 1.0)
    G, bs = res["Gamma"], list(res["beta_star"])
    rebuilt = [sum(G[i][k] * bs[k] for k in range(2)) for i in range(2)]
    assert list(res["beta"]) == pytest.approx(rebuilt, rel=1e-9)


def test_the_rotated_design_is_the_design_times_the_eigenvectors():
    # X* = X Gamma
    res = penfreg(Y, X, P, 1.0)
    G = res["Gamma"]
    for i in range(2):
        for j in range(2):
            expected = sum(X[i][k] * G[k][j] for k in range(2))
            assert res["X_star"][i][j] == pytest.approx(expected, rel=1e-9)


def test_the_penalty_matrix_eigenvalues_become_the_diagonal_penalty():
    res = penfreg(Y, X, [[3.0, 0.0], [0.0, 5.0]], 1.0)
    assert sorted(res["eigenvalues"]) == pytest.approx([3.0, 5.0],
                                                        rel=1e-9)


def test_shrinkage_pulls_the_coefficients_towards_zero():
    free = penfreg(Y, X, P, 0.0)["beta"]
    tied = penfreg(Y, X, P, 10.0)["beta"]
    assert sum(abs(b) for b in tied) < sum(abs(b) for b in free)
