"""Verification tests for msm323.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 15, eq. 15.1 p.497, the zero-altered Poisson links. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm323 import mvsml_functional_regression_eq_15_1


def test_the_count_link_is_inverted_by_exponentiating():
    # log(mu) = f_mu(x), so mu is exp of the forest prediction
    res = mvsml_functional_regression_eq_15_1(2.0, 0.25)
    assert res["mu"] == pytest.approx(math.exp(2.0), rel=1e-12)
    assert res["estimate"] == pytest.approx(math.exp(2.0), rel=1e-12)


def test_the_zero_link_is_inverted_by_the_logistic_function():
    # log(theta / (1 - theta)) = f_theta(x)
    res = mvsml_functional_regression_eq_15_1(2.0, 0.25)
    assert res["theta"] == pytest.approx(
        1.0 / (1.0 + math.exp(-0.25)), rel=1e-12)


def test_a_zero_linear_predictor_gives_a_unit_mean_and_an_even_chance():
    res = mvsml_functional_regression_eq_15_1(0.0, 0.0)
    assert res["mu"] == pytest.approx(1.0, rel=1e-12)
    assert res["theta"] == pytest.approx(0.5, rel=1e-12)


def test_the_zero_probability_stays_inside_the_unit_interval():
    for f in (-50.0, -1.0, 0.0, 1.0, 50.0):
        th = mvsml_functional_regression_eq_15_1(0.0, f)["theta"]
        assert 0.0 <= th <= 1.0
