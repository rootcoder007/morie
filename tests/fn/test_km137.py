"""Verification tests for km137.

Kamath, Keenan, Somers and Sorenson (2024), eq. 9.9, image-text matching with hard negatives. Expected values are
recomputed in the test body, and the value the docstring quotes is
asserted as well.
"""

import math

import pytest

from morie.fn.km137 import kamath_ch9_itm_hard_negative


def test_the_hard_negative_loss_sums_over_both_sets():
    # L_ITM-hn = -sum_Pos log p(aligned) - sum_HardNeg log p(unaligned)
    res = kamath_ch9_itm_hard_negative([0.5, 0.5], [0.5])
    assert res["estimate"] == pytest.approx(3.0 * math.log(2.0), rel=1e-12)


def test_adding_a_hard_negative_raises_the_loss():
    fewer = kamath_ch9_itm_hard_negative([0.5], [0.5])["estimate"]
    more = kamath_ch9_itm_hard_negative([0.5], [0.5, 0.5])["estimate"]
    assert more > fewer


def test_certainty_costs_nothing():
    res = kamath_ch9_itm_hard_negative([1.0], [1.0])
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)
