"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e58.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_58."""

import math

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e58 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_58,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e58_basic():
    """Test basic functionality."""
    n = 100
    p = 0.5
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_58(n, p)
    assert isinstance(result, dict)
    assert "sd_single" in result
    assert "sd_tot" in result
    assert "sd_avg" in result
    assert math.isfinite(result["sd_single"])
    assert math.isfinite(result["sd_tot"])
    assert math.isfinite(result["sd_avg"])
    assert result["sd_single"] >= 0
    assert result["sd_tot"] >= 0
    assert result["sd_avg"] >= 0


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e58_edge():
    """Test edge cases."""
    n = 50
    p = 0.1
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_58(n, p)
    assert isinstance(result, dict)
    assert "sd_single" in result
    assert "sd_tot" in result
    assert "sd_avg" in result
    assert math.isfinite(result["sd_single"])
    assert math.isfinite(result["sd_tot"])
    assert math.isfinite(result["sd_avg"])
    assert result["sd_single"] >= 0
    assert result["sd_tot"] >= 0
    assert result["sd_avg"] >= 0
