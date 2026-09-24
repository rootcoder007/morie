"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e45.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_45."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e45 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_45,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e45_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_45(1.0, 100)
    assert isinstance(result, dict)
    assert "sigma" in result
    assert "n" in result
    assert "sd_sum" in result
    assert math.isfinite(result["sd_sum"])
    assert result["n"] == 100


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e45_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_45(0.5, 10)
    assert isinstance(result, dict)
    assert "sd_sum" in result
    assert math.isfinite(result["sd_sum"])
    assert result["sigma"] == 0.5
    assert result["n"] == 10
