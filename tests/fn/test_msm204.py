"""Verification tests for msm204.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 9, eq. 9.30 p.348, KKT complementary slackness. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm204 import mvsml_ridge_lasso_elastic_eq_9_30


X = [[1.0, 1.0], [3.0, 3.0], [-1.0, -1.0]]
Y = [1, 1, -1]
B0, B = -1.0, [1.0, 1.0]


def test_the_margin_slack_is_the_functional_margin_less_one():
    # f(x) = -1 + x_1 + x_2 gives 1, 5 and -3, so y f = 1, 5, 3
    res = mvsml_ridge_lasso_elastic_eq_9_30([0.5, 0.0, 0.0], X, Y, B0, B)
    assert list(res["margin_slack"]) == pytest.approx([0.0, 4.0, 2.0],
                                                       abs=1e-12)


def test_slackness_holds_when_only_the_point_on_the_margin_is_weighted():
    res = mvsml_ridge_lasso_elastic_eq_9_30([0.5, 0.0, 0.0], X, Y, B0, B)
    assert list(res["complementary_products"]) == pytest.approx(
        [0.0, 0.0, 0.0], abs=1e-12)
    assert res["satisfied"] is True
    assert list(res["on_margin"]) == [0]


def test_weighting_a_point_off_the_margin_breaks_the_condition():
    # eq 9.30 forces alpha_i = 0 whenever y_i f(x_i) > 1
    res = mvsml_ridge_lasso_elastic_eq_9_30([0.5, 0.7, 0.0], X, Y, B0, B)
    assert res["satisfied"] is False
    assert res["complementary_products"][1] == pytest.approx(0.7 * 4.0,
                                                              rel=1e-12)


def test_a_support_vector_sits_exactly_on_the_unit_margin():
    res = mvsml_ridge_lasso_elastic_eq_9_30([0.5, 0.0, 0.0], X, Y, B0, B)
    i = res["on_margin"][0]
    f_i = B0 + sum(a * b for a, b in zip(X[i], B))
    assert Y[i] * f_i == pytest.approx(1.0, rel=1e-12)
