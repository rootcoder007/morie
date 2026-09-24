"""Tests for km050.kamath_ch3_back_translation_prob."""

from morie.fn import _array_core as np

from morie.fn.km050 import kamath_ch3_back_translation_prob

import math


def test_km050_basic():
    """Test basic functionality."""
    t = "[z] is the capital of [x]."
    thatt = "[z] est la capitale de [x]."
    p_forward = 0.5
    p_backward = 0.25
    result = kamath_ch3_back_translation_prob(t, thatt, p_forward=p_forward, p_backward=p_backward)

    # The estimate must be the product of the two supplied leg probabilities
    expected_estimate = p_forward * p_backward
    assert math.isclose(result["estimate"], expected_estimate)

    # The result should preserve the inputs and contain the expected keys
    assert result["p_forward"] == p_forward
    assert result["p_backward"] == p_backward
    assert result["candidate"] == t
    assert result["pivot"] == thatt
    assert result["n"] == 2

    # The estimate is a probability and must lie in [0, 1]
    assert 0.0 <= result["estimate"] <= 1.0


def test_km050_edge():
    """Test edge cases with extreme probabilities."""
    t = "prompt"
    thatt = "pivot"

    # Both probabilities zero -> estimate should be zero
    p_forward = 0.0
    p_backward = 0.0
    result = kamath_ch3_back_translation_prob(t, thatt, p_forward=p_forward, p_backward=p_backward)
    assert math.isclose(result["estimate"], 0.0)
    assert result["p_forward"] == p_forward
    assert result["p_backward"] == p_backward

    # Both probabilities one -> estimate should be one
    p_forward = 1.0
    p_backward = 1.0
    result = kamath_ch3_back_translation_prob(t, thatt, p_forward=p_forward, p_backward=p_backward)
    assert math.isclose(result["estimate"], 1.0)
    assert result["p_forward"] == p_forward
    assert result["p_backward"] == p_backward
