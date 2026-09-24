"""Verification tests for grn001.

Geron (2023), *Hands-On Machine Learning with Scikit-Learn, Keras and
TensorFlow*, 3rd edition, ch 4, the fitted life satisfaction line. Expected values are recomputed in
the test body and the docstring's own worked value is asserted too.
"""

import math

import pytest

from morie.fn.grn001 import geron_ch4_simple_linear_life_satisfaction


def test_the_line_is_the_intercept_plus_the_scaled_covariate():
    res = geron_ch4_simple_linear_life_satisfaction(4.85, 4.91e-5, 20000.0)
    assert res["life_satisfaction"] == pytest.approx(
        4.85 + 4.91e-5 * 20000.0, rel=1e-12)
    assert round(res["life_satisfaction"], 4) == pytest.approx(5.832,
                                                                abs=1e-4)


def test_doubling_the_covariate_doubles_only_the_slope_term():
    one = geron_ch4_simple_linear_life_satisfaction(4.85, 4.91e-5, 20000.0)["life_satisfaction"]
    two = geron_ch4_simple_linear_life_satisfaction(4.85, 4.91e-5, 40000.0)["life_satisfaction"]
    assert two - one == pytest.approx(4.91e-5 * 20000.0, rel=1e-12)
    assert round(two - one, 4) == pytest.approx(0.982, abs=1e-4)


def test_a_zero_covariate_leaves_only_the_intercept():
    assert geron_ch4_simple_linear_life_satisfaction(4.85, 4.91e-5, 0.0)["life_satisfaction"] == pytest.approx(
        4.85, rel=1e-12)


def test_a_zero_slope_makes_the_prediction_flat():
    a = geron_ch4_simple_linear_life_satisfaction(4.85, 0.0, 0.0)["life_satisfaction"]
    b = geron_ch4_simple_linear_life_satisfaction(4.85, 0.0, 1e6)["life_satisfaction"]
    assert a == pytest.approx(b, rel=1e-12)
