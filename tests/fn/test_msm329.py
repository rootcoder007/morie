"""Verification tests for msm329.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 15, eq. 15.4 p.498, the thresholded prediction. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm329 import mvsml_functional_regression_eq_15_4


def test_a_zero_probability_below_the_threshold_keeps_the_count_mean():
    res = mvsml_functional_regression_eq_15_4(0.25, 2.0)
    assert res["estimate"] == pytest.approx(2.0, rel=1e-12)
    assert res["is_zero"] is False


def test_a_zero_probability_above_the_threshold_predicts_a_zero():
    res = mvsml_functional_regression_eq_15_4(0.75, 2.0)
    assert res["estimate"] == pytest.approx(0.0, abs=1e-12)
    assert res["is_zero"] is True


def test_the_default_threshold_is_one_half_because_nothing_is_assumed():
    assert mvsml_functional_regression_eq_15_4(0.5000001, 2.0)["is_zero"] is True
    assert mvsml_functional_regression_eq_15_4(0.4999999, 2.0)["is_zero"] is False


def test_a_moved_threshold_moves_the_decision():
    assert mvsml_functional_regression_eq_15_4(0.6, 2.0, threshold=0.7)["is_zero"] is False
    assert mvsml_functional_regression_eq_15_4(0.6, 2.0, threshold=0.5)["is_zero"] is True
