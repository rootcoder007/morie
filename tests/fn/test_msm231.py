"""Verification tests for msm231.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 9, eqs. 9.44 and 9.45 p.354, the soft margin dual. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm231 import svmsdual


X = [[1.0, 1.0], [-1.0, -1.0]]
Y = [1, -1]


def _dual(alpha):
    lin = sum(alpha)
    quad = 0.0
    for i in range(len(alpha)):
        for j in range(len(alpha)):
            dot = sum(a * b for a, b in zip(X[i], X[j]))
            quad += alpha[i] * alpha[j] * Y[i] * Y[j] * dot
    return lin - 0.5 * quad


def test_the_dual_objective_is_the_linear_less_half_the_quadratic_form():
    res = svmsdual(X, Y, 1.0)
    assert res["objective"] == pytest.approx(_dual(list(res["alpha"])),
                                              rel=1e-9)


def test_the_optimum_of_this_two_point_problem_is_a_quarter_each():
    # x_i . x_j is 2 on the diagonal and -2 off it, so the objective is
    # 2a - 4a^2, maximised at a = 1/4 with value 1/4
    res = svmsdual(X, Y, 1.0)
    assert list(res["alpha"]) == pytest.approx([0.25, 0.25], rel=1e-6)
    assert res["objective"] == pytest.approx(0.25, rel=1e-6)


def test_the_multipliers_respect_both_constraints_of_equation_9_45():
    res = svmsdual(X, Y, 1.0)
    assert res["balance"] == pytest.approx(0.0, abs=1e-9)
    assert all(-1e-9 <= a <= 1.0 + 1e-9 for a in res["alpha"])
    assert res["bounded"] is True


def test_the_coefficients_follow_from_the_multipliers():
    res = svmsdual(X, Y, 1.0)
    rebuilt = [sum(a * y * xi[j] for a, y, xi in zip(res["alpha"], Y, X))
               for j in range(2)]
    assert list(res["beta"]) == pytest.approx(rebuilt, rel=1e-9)


def test_a_tighter_budget_caps_the_multipliers():
    res = svmsdual(X, Y, 0.1)
    assert all(a <= 0.1 + 1e-9 for a in res["alpha"])
