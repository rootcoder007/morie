"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e89.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_89."""

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e89 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_89,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e89_basic():
    """Test basic functionality."""
    # Poisson mode requires a scalar rate parameter a (lambda >= 0)
    a = 5.0
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_89(a)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result or "mode" in result or "k_star" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e89_edge():
    """Test edge cases."""
    # For a < 1, the Poisson mode is 0
    a = 0.5
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_89(a)
    assert isinstance(result, dict)
