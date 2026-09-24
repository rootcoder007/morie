"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e93.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_93."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e93 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_93,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e93_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_93(1.0, 100)
    assert isinstance(result, dict)
    assert "sd_mean" in result
    assert "sigma" in result
    assert math.isfinite(result["sd_mean"])
    assert result["sd_mean"] > 0


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e93_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_93(2.5, 2)
    assert isinstance(result, dict)
    assert "sd_mean" in result
    assert "sigma" in result
    assert math.isfinite(result["sd_mean"])
    assert result["sd_mean"] > 0
