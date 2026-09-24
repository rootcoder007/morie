"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e48.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_48."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e48 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_48,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e48_basic():
    """Test basic functionality."""
    n = 40
    with np.errstate(invalid="ignore"):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_48(n)
    assert isinstance(result, dict)
    assert "sd_tot" in result
    assert math.isfinite(float(result["sd_tot"]))


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e48_edge():
    """Test edge cases."""
    n = 1
    with np.errstate(invalid="ignore"):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_48(n)
    assert isinstance(result, dict)
    assert "sd_tot" in result
    assert math.isfinite(float(result["sd_tot"]))
