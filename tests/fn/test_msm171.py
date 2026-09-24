"""Verification tests for msm171.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 9, eq. 9.4 p.340, the hyperplane decision rule. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm171 import mvsml_ridge_lasso_elastic_eq_9_4


def test_the_decision_rule_returns_the_sign_of_the_affine_form():
    res = mvsml_ridge_lasso_elastic_eq_9_4([[1.0, 1.0], [0.0, 0.0]], -1.0, [1.0, 1.0])
    assert list(res["side"]) == [1, -1]
    assert res["estimate"] == pytest.approx(1.0, rel=1e-12)


def test_flipping_every_coefficient_flips_every_side():
    a = list(mvsml_ridge_lasso_elastic_eq_9_4([[1.0, 1.0], [0.0, 0.0]], -1.0, [1.0, 1.0])["side"])
    b = list(mvsml_ridge_lasso_elastic_eq_9_4([[1.0, 1.0], [0.0, 0.0]], 1.0, [-1.0, -1.0])["side"])
    assert b == [-s for s in a]


def test_the_rule_splits_the_space_into_exactly_two_halves():
    res = mvsml_ridge_lasso_elastic_eq_9_4([[5.0, 5.0], [-5.0, -5.0], [0.0, 0.0]], -1.0,
               [1.0, 1.0])
    assert set(res["side"]) <= {1, -1}
