"""Verification tests for msm201.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 9, eqs. 9.27 to 9.29 p.347, the hard margin Lagrangian. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm201 import svmlagr


X = [[1.0, 1.0], [-1.0, -1.0]]
Y = [1, -1]
BETA = [0.5, 0.5]
ALPHA = [0.25, 0.25]


def test_the_lagrangian_is_the_half_norm_less_the_weighted_slack():
    # eq 9.27: L = ||beta||^2 / 2 - sum alpha_i [y_i(beta_0 + x_i beta) - 1]
    res = svmlagr(X, Y, 0.0, BETA, ALPHA)
    brackets = [y * sum(a * b for a, b in zip(xi, BETA)) - 1.0
                for xi, y in zip(X, Y)]
    expected = 0.5 * sum(b * b for b in BETA) \
        - sum(a * br for a, br in zip(ALPHA, brackets))
    assert res["L"] == pytest.approx(expected, rel=1e-12)
    assert res["L"] == pytest.approx(0.25, rel=1e-12)


def test_both_points_sit_exactly_on_the_unit_margin():
    res = svmlagr(X, Y, 0.0, BETA, ALPHA)
    assert list(res["slack"]) == pytest.approx([0.0, 0.0], abs=1e-12)


def test_the_coefficient_gradient_vanishes_at_the_optimum():
    # eq 9.28: beta = sum_i alpha_i y_i x_i
    res = svmlagr(X, Y, 0.0, BETA, ALPHA)
    rebuilt = [sum(a * y * xi[j] for a, y, xi in zip(ALPHA, Y, X))
               for j in range(2)]
    assert rebuilt == pytest.approx(BETA, rel=1e-12)
    assert list(res["grad_beta"]) == pytest.approx([0.0, 0.0], abs=1e-12)


def test_the_intercept_gradient_is_the_multiplier_label_balance():
    # eq 9.29: sum_i alpha_i y_i = 0
    res = svmlagr(X, Y, 0.0, BETA, ALPHA)
    assert res["grad_beta0"] == pytest.approx(
        -sum(a * y for a, y in zip(ALPHA, Y)), abs=1e-12)
    assert res["grad_beta0"] == pytest.approx(0.0, abs=1e-12)


def test_unbalanced_multipliers_leave_a_gradient_behind():
    res = svmlagr(X, Y, 0.0, BETA, [0.5, 0.25])
    assert abs(res["grad_beta0"]) > 1e-9
