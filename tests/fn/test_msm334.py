"""Verification tests for msm334.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 3, sec. 3.5 p.80, the expected prediction error. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm334 import mvsml_preprocessing_eq_2_22


def test_the_error_is_the_noise_times_one_plus_the_scaled_projections():
    # EPE(x_o) = sigma2 (1 + sum_j (x*_oj)^2 / lambda_j)
    res = mvsml_preprocessing_eq_2_22(2.0, [1.0, 2.0], [1.0, 4.0])
    expected = 2.0 * (1.0 + 1.0 / 1.0 + 4.0 / 4.0)
    assert res["estimate"] == pytest.approx(expected, rel=1e-12)
    assert res["estimate"] == pytest.approx(6.0, rel=1e-12)


def test_the_irreducible_part_is_the_noise_variance_itself():
    res = mvsml_preprocessing_eq_2_22(2.0, [1.0, 2.0], [1.0, 4.0])
    assert res["irreducible"] == pytest.approx(2.0, rel=1e-12)
    assert res["variance_inflation"] == pytest.approx(3.0, rel=1e-12)


def test_a_point_at_the_origin_costs_only_the_irreducible_noise():
    res = mvsml_preprocessing_eq_2_22(1.5, [0.0, 0.0], [1.0, 4.0])
    assert res["estimate"] == pytest.approx(1.5, rel=1e-12)
    assert res["variance_inflation"] == pytest.approx(1.0, rel=1e-12)


def test_a_nearly_dependent_feature_blows_the_error_up():
    # a small eigenvalue is what makes collinearity expensive
    tame = mvsml_preprocessing_eq_2_22(1.0, [1.0], [1.0])["estimate"]
    wild = mvsml_preprocessing_eq_2_22(1.0, [1.0], [1e-6])["estimate"]
    assert wild > 1e5 * tame


def test_the_error_grows_with_the_noise_in_proportion():
    a = mvsml_preprocessing_eq_2_22(1.0, [1.0, 2.0], [1.0, 4.0])["estimate"]
    b = mvsml_preprocessing_eq_2_22(3.0, [1.0, 2.0], [1.0, 4.0])["estimate"]
    assert b == pytest.approx(3.0 * a, rel=1e-12)
