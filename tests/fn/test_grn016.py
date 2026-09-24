"""Verification tests for grn016.

Geron (2023), *Hands-On Machine Learning with Scikit-Learn, Keras and
TensorFlow*, 3rd edition, ch 4, the logistic regression decision rule. Expected values are recomputed in
the test body and the docstring's own worked value is asserted too.
"""

import math

import pytest

from morie.fn.grn016 import geron_ch4_logistic_regression_prediction


def test_the_rule_labels_a_probability_at_or_above_one_half_positive():
    res = geron_ch4_logistic_regression_prediction([0.2, 0.5, 0.9])
    assert list(res["y_hat"]) == [0, 1, 1]
    assert round(res["positive_rate"], 10) == pytest.approx(
        2.0 / 3.0, abs=1e-10)


def test_exactly_one_half_goes_positive_rather_than_negative():
    assert geron_ch4_logistic_regression_prediction(0.5)["y_hat"] == 1


def test_a_moved_threshold_moves_the_labels():
    assert list(geron_ch4_logistic_regression_prediction([0.6], threshold=0.7)["y_hat"]) == [0]
    assert list(geron_ch4_logistic_regression_prediction([0.6], threshold=0.5)["y_hat"]) == [1]


def test_the_positive_rate_is_the_share_of_positive_labels():
    res = geron_ch4_logistic_regression_prediction([0.1, 0.2, 0.8, 0.9])
    assert res["positive_rate"] == pytest.approx(0.5, rel=1e-12)
