"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e95.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_95."""

import math
import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e95 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_95,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e95_basic():
    """Test basic functionality."""
    n, p = 20, 0.4
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_95(n, p)
    assert isinstance(result, dict)
    assert "k" in result
    assert "PP" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e95_edge():
    """Test edge cases."""
    n, p = 5, 0.0
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_95(n, p)
    assert isinstance(result, dict)
    assert "k" in result
    assert "PP" in result
