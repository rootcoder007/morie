"""Verification tests for msm327.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 15, eq. 15.3 p.498, the zero-altered Poisson mean. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm327 import mvsml_functional_regression_eq_15_3


def test_the_prediction_is_the_zero_altered_poisson_mean():
    # Y-hat = (1 - theta) mu / (1 - exp(-mu))
    res = mvsml_functional_regression_eq_15_3(0.25, 2.0)
    expected = (1.0 - 0.25) * 2.0 / (1.0 - math.exp(-2.0))
    assert res["estimate"] == pytest.approx(expected, rel=1e-12)
    assert res["y_hat"] == pytest.approx(expected, rel=1e-12)


def test_a_certain_zero_predicts_nothing_at_all():
    res = mvsml_functional_regression_eq_15_3(1.0, 2.0)
    assert res["estimate"] == pytest.approx(0.0, abs=1e-12)


def test_the_truncation_factor_lifts_the_mean_above_the_poisson_one():
    # dividing by 1 - exp(-mu) is the zero-truncation correction, so the
    # result exceeds (1 - theta) mu
    res = mvsml_functional_regression_eq_15_3(0.0, 2.0)
    assert res["estimate"] > 2.0


def test_a_larger_zero_probability_lowers_the_prediction():
    low = mvsml_functional_regression_eq_15_3(0.1, 2.0)["estimate"]
    high = mvsml_functional_regression_eq_15_3(0.9, 2.0)["estimate"]
    assert high < low
