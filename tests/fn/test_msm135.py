"""Verification tests for msm135.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 8, eq. 8.6 p.276, the RKHS estimating equations. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm135 import mvsml_categorical_count_eq_8_6


C = [[1.0], [1.0]]
K = [[2.0, 0.0], [0.0, 3.0]]
Y = [1.0, 3.0]


def _matvec(M, v):
    return [sum(a * b for a, b in zip(row, v)) for row in M]


def test_the_solution_satisfies_the_first_block_of_equation_8_6():
    # C'C theta + C'K beta = C'y
    res = mvsml_categorical_count_eq_8_6(C, K, Y, lam=1.0, sigma2_e=1.0)
    th, be = list(res["theta"]), list(res["beta"])
    lhs = sum(sum(c) * t for c, t in zip(zip(*C), th))
    lhs = sum(C[i][0] * sum(C[i][j] * th[j] for j in range(len(th)))
              for i in range(len(C)))
    lhs += sum(C[i][0] * sum(K[i][j] * be[j] for j in range(len(be)))
               for i in range(len(C)))
    rhs = sum(C[i][0] * Y[i] for i in range(len(C)))
    assert lhs == pytest.approx(rhs, rel=1e-9)


def test_the_solution_satisfies_the_second_block_of_equation_8_6():
    # K'C theta + (K'K + lambda K sigma2_e) beta = K'y
    lam, s2 = 1.0, 1.0
    res = mvsml_categorical_count_eq_8_6(C, K, Y, lam=lam, sigma2_e=s2)
    th, be = list(res["theta"]), list(res["beta"])
    n = len(Y)
    for i in range(n):
        lhs = sum(K[r][i] * sum(C[r][j] * th[j] for j in range(len(th)))
                  for r in range(n))
        lhs += sum((sum(K[r][i] * K[r][j] for r in range(n))
                    + lam * K[i][j] * s2) * be[j] for j in range(n))
        rhs = sum(K[r][i] * Y[r] for r in range(n))
        assert lhs == pytest.approx(rhs, rel=1e-9)


def test_the_breeding_values_are_the_kernel_times_the_coefficients():
    res = mvsml_categorical_count_eq_8_6(C, K, Y, lam=1.0, sigma2_e=1.0)
    be = list(res["beta"])
    assert list(res["u"]) == pytest.approx(_matvec(K, be), rel=1e-9)


def test_the_marker_variance_is_the_reciprocal_of_the_penalty():
    res = mvsml_categorical_count_eq_8_6(C, K, Y, lam=4.0, sigma2_e=1.0)
    assert res["sigma2_beta"] == pytest.approx(0.25, rel=1e-12)
