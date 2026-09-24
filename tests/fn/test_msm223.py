"""Verification tests for msm223.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 9, eqs. 9.38 to 9.43 pp.352-353, the Wolfe primal. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm223 import svmkkt


X = [[1.0, 1.0], [-1.0, -1.0]]
Y = [1, -1]
ALPHA = [0.25, 0.25]
DELTA = [0.75, 0.75]
ZETA = [0.0, 0.0]
BETA = [0.5, 0.5]
TT = 1.0


def test_the_primal_objective_is_the_norm_plus_the_penalised_slack():
    # eq 9.38: L = ||beta||^2 / 2 + T sum zeta - sum alpha [y f - 1 + zeta]
    #              - sum delta zeta, and at this optimum every bracket is 0
    res = svmkkt(X, Y, 0.0, BETA, ALPHA, DELTA, ZETA, TT)
    assert res["L"] == pytest.approx(0.5 * (0.5 ** 2 + 0.5 ** 2),
                                      rel=1e-12)
    assert res["L"] == pytest.approx(0.25, rel=1e-12)


def test_the_coefficients_are_the_weighted_sum_of_the_support_vectors():
    # eq 9.39: beta = sum_i alpha_i y_i x_i
    rebuilt = [sum(a * y * xi[j] for a, y, xi in zip(ALPHA, Y, X))
               for j in range(2)]
    assert rebuilt == pytest.approx(BETA, rel=1e-12)
    res = svmkkt(X, Y, 0.0, BETA, ALPHA, DELTA, ZETA, TT)
    assert list(res["stationarity_beta"]) == pytest.approx([0.0, 0.0],
                                                            abs=1e-12)


def test_the_multipliers_balance_against_the_labels():
    # eq 9.40: sum_i alpha_i y_i = 0
    res = svmkkt(X, Y, 0.0, BETA, ALPHA, DELTA, ZETA, TT)
    assert res["balance"] == pytest.approx(
        sum(a * y for a, y in zip(ALPHA, Y)), abs=1e-12)
    assert res["balance"] == pytest.approx(0.0, abs=1e-12)


def test_each_pair_of_multipliers_sums_to_the_budget():
    # eq 9.41: alpha_i + delta_i = T
    res = svmkkt(X, Y, 0.0, BETA, ALPHA, DELTA, ZETA, TT)
    assert list(res["multiplier_sum"]) == pytest.approx([0.0, 0.0],
                                                         abs=1e-12)
    for a, d in zip(ALPHA, DELTA):
        assert a + d == pytest.approx(TT, rel=1e-12)


def test_both_slackness_conditions_hold_at_the_optimum():
    # eqs 9.42 and 9.43
    res = svmkkt(X, Y, 0.0, BETA, ALPHA, DELTA, ZETA, TT)
    assert list(res["complementary_alpha"]) == pytest.approx([0.0, 0.0],
                                                              abs=1e-12)
    assert list(res["complementary_delta"]) == pytest.approx([0.0, 0.0],
                                                              abs=1e-12)
    assert res["kkt_satisfied"] is True


def test_a_budget_that_contradicts_the_multipliers_is_reported():
    res = svmkkt(X, Y, 0.0, BETA, ALPHA, DELTA, ZETA, 5.0)
    assert res["kkt_satisfied"] is False
