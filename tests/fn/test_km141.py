"""Verification tests for km141.

Kamath, Keenan, Somers and Sorenson (2024), the image-text matching loss. Expected values are
recomputed in the test body, and the value the docstring quotes is
asserted as well.
"""

import math

import pytest

from morie.fn.km141 import kamath_ch9_itm_loss


def test_the_matching_loss_is_the_mean_binary_cross_entropy():
    # L_ITM = -E[y log s + (1-y) log(1-s)]
    res = kamath_ch9_itm_loss([0.9, 0.2], None, None, [1, 0])
    expected = (-math.log(0.9) - math.log(0.8)) / 2.0
    assert res["estimate"] == pytest.approx(expected, rel=1e-12)


def test_a_perfect_classifier_costs_nothing():
    res = kamath_ch9_itm_loss([1.0, 0.0], None, None, [1, 0])
    assert res["estimate"] == pytest.approx(0.0, abs=1e-12)


def test_an_inverted_classifier_is_expensive():
    poor = kamath_ch9_itm_loss([0.1, 0.9], None, None, [1, 0])["estimate"]
    good = kamath_ch9_itm_loss([0.9, 0.1], None, None, [1, 0])["estimate"]
    assert poor > good
