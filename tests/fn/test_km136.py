"""Verification tests for km136.

Kamath, Keenan, Somers and Sorenson (2024), the multimodal vision-language loss. Expected values are
recomputed in the test body, and the value the docstring quotes is
asserted as well.
"""

import math

import pytest

from morie.fn.km136 import kamath_ch9_mml_vlm_loss


def test_the_vision_language_loss_sums_both_halves():
    # L = -sum_Pos log p(aligned) - sum_Neg log p(unaligned)
    res = kamath_ch9_mml_vlm_loss([0.9], [0.8])
    assert res["estimate"] == pytest.approx(
        -math.log(0.9) - math.log(0.8), rel=1e-12)


def test_certainty_on_both_sides_costs_nothing():
    res = kamath_ch9_mml_vlm_loss([1.0], [1.0])
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_the_loss_grows_as_the_model_gets_less_certain():
    good = kamath_ch9_mml_vlm_loss([0.9], [0.9])["estimate"]
    poor = kamath_ch9_mml_vlm_loss([0.4], [0.4])["estimate"]
    assert poor > good
