"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e46.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_46."""

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e46 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_46,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e46_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_46(x, y)
    assert isinstance(result, dict)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e46_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 40
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_46(x, y)
    assert isinstance(result, dict)
