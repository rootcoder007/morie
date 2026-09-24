"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner1e17.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_17."""

import math

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner1e17 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_17,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e17_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_17(5, 3)
    assert isinstance(result, dict)
    assert "n_picks" in result
    assert "n_types" in result
    assert "count" in result
    assert math.isfinite(result["count"])
    assert result["count"] > 0
    assert result["n_picks"] == 5
    assert result["n_types"] == 3


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e17_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_17(1, 1)
    assert isinstance(result, dict)
    assert "count" in result
    assert result["count"] == 1
    assert result["n_picks"] == 1
    assert result["n_types"] == 1
