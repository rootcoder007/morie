"""Verification tests for msm137.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 8, eq. 8.7 p.276, the reduced RKHS equations. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm137 import mvsml_categorical_count_eq_8_7


C = [[1.0], [1.0]]
K = [[2.0, 0.0], [0.0, 3.0]]
Y = [1.0, 3.0]


def test_the_reduced_form_reaches_the_same_solution_as_equation_8_6():
    # the book states that multiplying the second block by K^-1 leaves
    # the solution unchanged
    from morie.fn.msm135 import mvsml_rkhs_mixed_equations as direct
    red = mvsml_categorical_count_eq_8_7(C, K, Y, lam=1.0, sigma2_e=1.0)
    dir_ = direct(C, K, Y, lam=1.0, sigma2_e=1.0)
    assert list(red["theta"]) == pytest.approx(list(dir_["theta"]),
                                               rel=1e-8)
    assert list(red["beta"]) == pytest.approx(list(dir_["beta"]),
                                              rel=1e-8)


def test_the_solution_satisfies_the_second_block_of_equation_8_7():
    # C theta + (K + lambda I sigma2_e) beta = y
    lam, s2 = 1.0, 1.0
    res = mvsml_categorical_count_eq_8_7(C, K, Y, lam=lam, sigma2_e=s2)
    th, be = list(res["theta"]), list(res["beta"])
    for i in range(len(Y)):
        lhs = sum(C[i][j] * th[j] for j in range(len(th)))
        lhs += sum((K[i][j] + (lam * s2 if i == j else 0.0)) * be[j]
                   for j in range(len(be)))
        assert lhs == pytest.approx(Y[i], rel=1e-9)


def test_new_genotypes_are_predicted_by_one_kernel_product():
    res = mvsml_categorical_count_eq_8_7(C, K, Y, lam=1.0, sigma2_e=1.0,
               K_star=[[2.0, 0.0]])
    be = list(res["beta"])
    assert list(res["u_new"]) == pytest.approx(
        [2.0 * be[0] + 0.0 * be[1]], rel=1e-9)
