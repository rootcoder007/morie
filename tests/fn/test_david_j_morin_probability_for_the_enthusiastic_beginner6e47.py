"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e47.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_47."""

import pytest
from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e47 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_47,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e47_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_47(x, y)
    assert isinstance(result, dict)
    assert "A" in result
    assert "B" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e47_edge():
    """Test edge cases."""
    rng = np.random.default_rng(7)
    n = 40
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_47(x, y)
    assert isinstance(result, dict)
    assert "A" in result
    assert "B" in result
