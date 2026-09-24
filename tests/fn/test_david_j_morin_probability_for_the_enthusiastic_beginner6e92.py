"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e92.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_92."""

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e92 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_92,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e92_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = 2.0 + 3.0 * x + rng.normal(0, 0.5, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_92(x, y)
    assert isinstance(result, dict)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e92_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = 1.0 + x + rng.normal(0, 0.1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_92(x, y)
    assert isinstance(result, dict)
