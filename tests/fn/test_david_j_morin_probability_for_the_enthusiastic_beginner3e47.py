"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e47.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_47."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e47 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_47,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e47_basic():
    """Test basic functionality."""
    n = 40
    p = 0.3
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_47(n, p)
    assert isinstance(result, dict)
    assert "sd" in result
    assert math.isfinite(float(result["sd"]))
    assert float(result["sd"]) >= 0
    assert result["n"] == n
    assert result["p"] == p


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e47_edge():
    """Test edge cases."""
    n = 1
    p = 0.5
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_47(n, p)
    assert isinstance(result, dict)
    assert "sd" in result
    assert math.isfinite(float(result["sd"]))
    assert float(result["sd"]) >= 0
    assert result["n"] == n
    assert result["p"] == p
