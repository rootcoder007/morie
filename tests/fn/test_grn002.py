"""Verification tests for grn002.

Geron (2023), *Hands-On Machine Learning with Scikit-Learn, Keras and
TensorFlow*, 3rd edition, ch 4, the linear regression prediction. Expected values are recomputed in
the test body and the docstring's own worked value is asserted too.
"""

import math

import pytest

from morie.fn.grn002 import geron_ch4_linear_regression_prediction


def test_the_prediction_is_the_bias_plus_the_weighted_features():
    res = geron_ch4_linear_regression_prediction([1.0, 2.0, 3.0], [3.0, 4.0])
    assert res["prediction"] == pytest.approx(
        1.0 + 2.0 * 3.0 + 3.0 * 4.0, rel=1e-12)
    assert res["prediction"] == pytest.approx(19.0, rel=1e-12)


def test_each_feature_contribution_is_reported_separately():
    res = geron_ch4_linear_regression_prediction([1.0, 2.0, 3.0], [3.0, 4.0])
    assert list(res["contributions"]) == pytest.approx([6.0, 12.0],
                                                        rel=1e-12)
    assert sum(res["contributions"]) + 1.0 == pytest.approx(
        res["prediction"], rel=1e-12)


def test_a_batch_of_instances_is_predicted_row_by_row():
    res = geron_ch4_linear_regression_prediction([0.0, 1.0], [[2.0], [5.0]])
    assert list(res["prediction"]) == pytest.approx([2.0, 5.0],
                                                     rel=1e-12)


def test_zero_weights_leave_only_the_bias():
    res = geron_ch4_linear_regression_prediction([7.0, 0.0, 0.0], [100.0, -100.0])
    assert res["prediction"] == pytest.approx(7.0, rel=1e-12)
