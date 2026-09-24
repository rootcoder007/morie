"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e37.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_37."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e37 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_37,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e37_basic():
    """Test basic functionality."""
    sigma_signal = 1.0
    sigma_noise = 2.0
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_37(sigma_signal, sigma_noise)
    assert isinstance(result, dict)
    assert "r" in result
    r = result["r"]
    assert isinstance(r, (int, float))
    assert math.isfinite(r)
    assert 0 <= r <= 1


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e37_edge():
    """Test edge cases."""
    sigma_signal = 0.5
    sigma_noise = 0.5
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_37(sigma_signal, sigma_noise)
    assert isinstance(result, dict)
    assert "r" in result
    r = result["r"]
    assert isinstance(r, (int, float))
    assert math.isfinite(r)
    assert 0 <= r <= 1
