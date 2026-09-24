"""Verification tests for msm234.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 9, eqs. 9.46 and 9.47 p.357, the kernel dual. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm234 import ksvmdual


X = [[1.0, 1.0], [-1.0, -1.0]]
Y = [1, -1]


def test_the_linear_kernel_reproduces_the_inner_product_dual():
    # the chapter's whole point: the dual touches the data only through
    # x_i . x_j, so a linear kernel must return the same solution
    from morie.fn.msm231 import svmsdual
    k = ksvmdual(X, Y, 1.0, kernel="linear")
    p = svmsdual(X, Y, 1.0)
    assert list(k["alpha"]) == pytest.approx(list(p["alpha"]), rel=1e-8)
    assert list(k["beta"]) == pytest.approx(list(p["beta"]), rel=1e-8)
    assert k["objective"] == pytest.approx(p["objective"], rel=1e-8)


def test_the_linear_gram_matrix_is_the_matrix_of_inner_products():
    res = ksvmdual(X, Y, 1.0, kernel="linear")
    for i in range(2):
        for j in range(2):
            dot = sum(a * b for a, b in zip(X[i], X[j]))
            assert res["K"][i][j] == pytest.approx(dot, rel=1e-12)


def test_the_gram_matrix_is_symmetric_for_every_kernel():
    for kern in ("linear", "gaussian", "polynomial", "exponential"):
        res = ksvmdual(X, Y, 1.0, kernel=kern, gamma=0.5)
        K = res["K"]
        for i in range(2):
            for j in range(2):
                assert K[i][j] == pytest.approx(K[j][i], rel=1e-12)


def test_the_multipliers_respect_both_constraints_of_equation_9_47():
    res = ksvmdual(X, Y, 1.0, kernel="linear")
    assert res["balance"] == pytest.approx(0.0, abs=1e-9)
    assert all(-1e-9 <= a <= 1.0 + 1e-9 for a in res["alpha"])
