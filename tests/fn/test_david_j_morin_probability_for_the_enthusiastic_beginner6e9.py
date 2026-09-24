"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e9.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_9."""

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e9 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_9,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e9_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0, 1, 100)
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_9(x, y)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e9_edge():
    """Test edge cases."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 20)
    y = rng.normal(0, 1, 20)
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_9(x, y)
    assert isinstance(result, dict)
    assert len(result) > 0
