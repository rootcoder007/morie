"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner19e2.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_19_equation_2."""

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner19e2 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_19_equation_2,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner19e2_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 40)
    y = rng.normal(0, 1, 40)
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_19_equation_2(x, y)
    assert isinstance(result, dict)


def test_david_j_morin_probability_for_the_enthusiastic_beginner19e2_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 5)
    y = rng.normal(0, 1, 5)
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_19_equation_2(x, y)
    assert isinstance(result, dict)
