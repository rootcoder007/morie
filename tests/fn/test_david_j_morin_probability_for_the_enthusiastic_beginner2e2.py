"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner2e2.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_2."""

import math

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner2e2 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_2,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e2_basic():
    """Test basic functionality."""
    p_a = 0.3
    p_b = 0.4
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_2(p_a, p_b)
    assert isinstance(result, dict)
    for key in ("ps", "p_and", "p_a", "p_b"):
        assert key in result
    assert math.isfinite(result["p_and"])
    assert 0.0 <= result["p_and"] <= 1.0
    assert math.isfinite(result["p_a"])
    assert math.isfinite(result["p_b"])


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e2_edge():
    """Test edge cases."""
    p_a = 0.0
    p_b = 1.0
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_2(p_a, p_b)
    assert isinstance(result, dict)
    for key in ("ps", "p_and", "p_a", "p_b"):
        assert key in result
    assert math.isfinite(result["p_and"])
    assert 0.0 <= result["p_and"] <= 1.0
    assert math.isfinite(result["p_a"])
    assert math.isfinite(result["p_b"])
