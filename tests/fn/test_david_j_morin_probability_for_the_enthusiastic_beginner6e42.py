"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e42.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_42."""

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e42 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_42,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e42_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_42(x, y)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e42_edge():
    """Test edge cases."""
    rng = np.random.default_rng(0)
    n = 5
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_42(x, y)
    assert isinstance(result, dict)
    assert len(result) > 0
