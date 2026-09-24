"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e14.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_14."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e14 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_14,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e14_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_14(x, y)
    assert isinstance(result, dict)
    assert len(result) > 0
    for value in result.values():
        assert math.isfinite(value)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e14_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 10)
    y = rng.normal(0, 1, 10)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_14(x, y)
    assert isinstance(result, dict)
