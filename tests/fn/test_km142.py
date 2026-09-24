"""Verification tests for km142.

Kamath, Keenan, Somers and Sorenson (2024), the image-text generation loss. Expected values are
recomputed in the test body, and the value the docstring quotes is
asserted as well.
"""

import math

import pytest

from morie.fn.km142 import kamath_ch9_itg_loss


def test_the_generation_loss_sums_the_log_of_each_sequence_product():
    # L_ITG = -sum log prod_t P(y_t | y_<t, x)
    res = kamath_ch9_itg_loss(None, [[0.5, 0.5], [0.25]])
    # 1/4 for the first sequence and 1/4 for the second
    assert res["estimate"] == pytest.approx(2.0 * math.log(4.0), rel=1e-12)


def test_a_certain_generation_costs_nothing():
    res = kamath_ch9_itg_loss(None, [[1.0, 1.0]])
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_each_sequence_contributes_independently():
    one = kamath_ch9_itg_loss(None, [[0.5]])["estimate"]
    two = kamath_ch9_itg_loss(None, [[0.5], [0.5]])["estimate"]
    assert two == pytest.approx(2.0 * one, rel=1e-12)
