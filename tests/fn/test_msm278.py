"""Verification tests for msm278.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 14, eq. 14.11 p.470, the roughness penalty matrix. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm278 import penmat


GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def test_the_penalty_matrix_is_symmetric():
    res = penmat(GRID, 3, p=2)
    P = res["P"]
    for i in range(3):
        for j in range(3):
            assert P[i][j] == pytest.approx(P[j][i], rel=1e-9, abs=1e-12)


def test_the_constant_basis_function_carries_no_roughness():
    # the second derivative of a constant is zero, so its whole row
    # of the penalty matrix vanishes
    res = penmat(GRID, 3, p=2)
    assert list(res["P"][0]) == pytest.approx([0.0] * 3, abs=1e-9)


def test_a_constant_coefficient_vector_has_zero_penalty():
    res = penmat(GRID, 3, p=2, beta=[1.0, 0.0, 0.0])
    assert res["J"] == pytest.approx(0.0, abs=1e-9)


def test_the_penalty_is_the_quadratic_form_of_the_matrix():
    beta = [0.0, 1.0, 0.5]
    res = penmat(GRID, 3, p=2, beta=beta)
    P = res["P"]
    expected = sum(beta[i] * P[i][j] * beta[j] for i in range(3)
                   for j in range(3))
    assert res["J"] == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_the_reported_order_and_width_match_the_request():
    res = penmat(GRID, 4, p=1)
    assert res["order"] == 1
    assert res["L1"] == 4
    assert len(res["P"]) == 4
