"""Verification tests for msm271.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 14, eqs. 14.7 and 14.8 p.469, the basis matrix. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm271 import basmat


GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def test_the_basis_matrix_has_one_row_per_time_and_one_column_per_basis():
    # eq 14.8: Psi is m x L2 with Psi[j][o] = psi_o(t_j)
    res = basmat(GRID, 4)
    assert res["m"] == len(GRID)
    assert res["L2"] == 4
    assert len(res["Psi"]) == len(GRID)
    assert all(len(row) == 4 for row in res["Psi"])


def test_the_cross_product_is_the_basis_matrix_times_its_transpose():
    res = basmat(GRID, 3)
    Psi, PtP = res["Psi"], res["PsiTPsi"]
    for i in range(3):
        for j in range(3):
            expected = sum(Psi[r][i] * Psi[r][j]
                           for r in range(len(GRID)))
            assert PtP[i][j] == pytest.approx(expected, rel=1e-12)


def test_the_cross_product_is_symmetric_as_a_gram_matrix_must_be():
    res = basmat(GRID, 4)
    PtP = res["PsiTPsi"]
    for i in range(4):
        for j in range(4):
            assert PtP[i][j] == pytest.approx(PtP[j][i], rel=1e-12)


def test_the_first_fourier_basis_function_is_constant_over_time():
    res = basmat(GRID, 3)
    col = [row[0] for row in res["Psi"]]
    for v in col:
        assert v == pytest.approx(col[0], rel=1e-12)


def test_a_denser_grid_adds_rows_without_adding_columns():
    dense = basmat([i / 20.0 for i in range(21)], 3)
    assert dense["m"] == 21
    assert dense["L2"] == 3
