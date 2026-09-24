"""Verification tests for msm173.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 9, eq. 9.5 p.341, the support vector fitting function. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm173 import mvsml_ridge_lasso_elastic_eq_9_5


def test_the_fitting_function_is_the_intercept_plus_the_inner_product():
    # eq 9.5: f(x_i) = beta_0 + x_i' beta
    res = mvsml_ridge_lasso_elastic_eq_9_5([[1.0, 2.0], [-1.0, -1.0]], 0.5, [1.0, 2.0])
    assert list(res["f"]) == pytest.approx(
        [0.5 + 1.0 + 4.0, 0.5 - 1.0 - 2.0], rel=1e-12)
    assert res["estimate"] == pytest.approx(5.5, rel=1e-12)


def test_the_label_is_the_sign_of_the_fitting_function():
    res = mvsml_ridge_lasso_elastic_eq_9_5([[1.0, 2.0], [-1.0, -1.0]], 0.5, [1.0, 2.0])
    assert list(res["labels"]) == [1, -1]


def test_a_separable_training_set_has_a_positive_margin_everywhere():
    X = [[2.0, 2.0], [-2.0, -2.0]]
    y = [1, -1]
    res = mvsml_ridge_lasso_elastic_eq_9_5(X, 0.0, [1.0, 1.0])
    for yi, fi in zip(y, res["f"]):
        assert yi * fi > 0


def test_a_larger_magnitude_means_more_confidence_in_the_assignment():
    res = mvsml_ridge_lasso_elastic_eq_9_5([[10.0, 10.0], [0.1, 0.1]], 0.0, [1.0, 1.0])
    assert abs(res["f"][0]) > abs(res["f"][1])
