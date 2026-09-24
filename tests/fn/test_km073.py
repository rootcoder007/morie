"""Verification tests for km073.

Kamath, Keenan, Somers and Sorenson (2024), Large Language Models: A
Deep Dive, eq. 5.9, the preference as a sigmoid of the reward margin. Expected values are recomputed in the test body.
"""

import math

import pytest

from morie.fn.km073 import kamath_ch5_pref_sigmoid_form


def test_preference_is_the_sigmoid_of_the_reward_margin():
    # Eq 5.9: p* = sigma(r*(y_w) - r*(y_l))
    r = {"w": 2.0, "l": 1.0}
    res = kamath_ch5_pref_sigmoid_form(r)
    assert res["margin"] == pytest.approx(1.0, rel=1e-12)
    assert res["estimate"] == pytest.approx(1.0 / (1.0 + math.exp(-1.0)), rel=1e-12)


def test_an_equal_margin_gives_an_even_preference():
    res = kamath_ch5_pref_sigmoid_form({"w": 1.5, "l": 1.5})
    assert res["estimate"] == pytest.approx(0.5, rel=1e-12)


def test_a_larger_margin_is_a_stronger_preference():
    small = kamath_ch5_pref_sigmoid_form({"w": 1.1, "l": 1.0})["estimate"]
    large = kamath_ch5_pref_sigmoid_form({"w": 5.0, "l": 1.0})["estimate"]
    assert 0.5 < small < large < 1.0
