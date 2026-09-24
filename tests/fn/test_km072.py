"""Verification tests for km072.

Kamath, Keenan, Somers and Sorenson (2024), eq. 5.8, the Bradley-Terry preference probability. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km072 import kamath_ch5_bradley_terry_pref


def test_bradley_terry_preference_is_the_softmax_over_two_rewards():
    # Eq 5.8: p* = exp(r_w)/(exp(r_w) + exp(r_l))
    r = {"w": 2.0, "l": 1.0}
    res = kamath_ch5_bradley_terry_pref(r, "w", "l")
    expected = math.exp(2.0) / (math.exp(2.0) + math.exp(1.0))
    assert res["estimate"] == pytest.approx(expected, rel=1e-12)
    assert res["p_reversed"] == pytest.approx(1.0 - expected, rel=1e-12)
    assert res["margin"] == pytest.approx(1.0, rel=1e-12)


def test_the_softmax_form_equals_the_sigmoid_of_the_margin():
    # dividing through by exp(r_w) turns Eq 5.8 into Eq 5.9
    r = {"w": 1.3, "l": -0.4}
    res = kamath_ch5_bradley_terry_pref(r, "w", "l")
    margin = 1.3 - (-0.4)
    assert res["estimate"] == pytest.approx(1.0 / (1.0 + math.exp(-margin)), rel=1e-12)


def test_equal_rewards_give_an_even_preference():
    res = kamath_ch5_bradley_terry_pref({"a": 0.9, "b": 0.9}, "a", "b")
    assert res["estimate"] == pytest.approx(0.5, rel=1e-12)
