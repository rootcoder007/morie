"""Verification tests for msm218.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 9, eqs. 9.34 to 9.37 pp.350-351, the soft margin. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm218 import softsvm


X = [[1.0, 1.0], [-1.0, -1.0]]
Y = [1, -1]


def test_the_margin_is_the_reciprocal_of_the_coefficient_norm():
    res = softsvm(X, Y, 1.0)
    nb = math.sqrt(sum(b * b for b in res["beta"]))
    assert res["norm_beta"] == pytest.approx(nb, rel=1e-12)
    assert res["margin"] == pytest.approx(1.0 / nb, rel=1e-12)


def test_the_normalised_direction_meets_the_unit_norm_constraint():
    # eq 9.35 is stated for the normalised coefficients, and the
    # returned beta is the unnormalised direction with margin 1/||beta||
    res = softsvm(X, Y, 1.0)
    nb = res["norm_beta"]
    unit = [b / nb for b in res["beta"]]
    assert sum(u * u for u in unit) == pytest.approx(1.0, rel=1e-12)


def test_every_point_clears_the_margin_under_the_normalised_direction():
    # eq 9.36: y_i (beta_0 + x_i' beta_unit) >= M (1 - zeta_i)
    res = softsvm(X, Y, 1.0)
    nb = res["norm_beta"]
    for xi, yi, zi in zip(X, Y, res["zeta"]):
        f = (res["beta0"] + sum(a * b for a, b in zip(xi, res["beta"]))) / nb
        assert yi * f >= res["margin"] * (1.0 - zi) - 1e-9


def test_the_slack_budget_of_equation_9_37_is_respected():
    res = softsvm(X, Y, 1.0)
    assert all(z >= -1e-12 for z in res["zeta"])
    assert res["slack_sum"] == pytest.approx(sum(res["zeta"]), abs=1e-12)


def test_a_separable_pair_needs_no_slack_at_all():
    res = softsvm(X, Y, 1.0)
    assert res["n_violating"] == 0
    assert res["n_misclassified"] == 0
