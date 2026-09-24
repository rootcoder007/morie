"""Verification tests for km050.

Kamath, Keenan, Somers and Sorenson (2024), eq. 3.9, the round-trip back-translation score. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km050 import kamath_ch3_back_translation_prob


def test_the_round_trip_probability_is_the_product_of_both_directions():
    # Eq 3.9: P(t) = P_forward(t_hat|t) P_backward(t|t_hat)
    for f, b in ((0.6, 0.5), (1.0, 0.25), (0.1, 0.1)):
        res = kamath_ch3_back_translation_prob("orig", "para", f, b)
        assert res["estimate"] == pytest.approx(f * b, rel=1e-12)


def test_a_certain_round_trip_scores_one():
    assert kamath_ch3_back_translation_prob("t", "t", 1.0, 1.0)["estimate"] == pytest.approx(1.0, rel=1e-12)


def test_a_broken_direction_kills_the_score():
    assert kamath_ch3_back_translation_prob("t", "p", 0.9, 0.0)["estimate"] == pytest.approx(0.0, abs=1e-15)
