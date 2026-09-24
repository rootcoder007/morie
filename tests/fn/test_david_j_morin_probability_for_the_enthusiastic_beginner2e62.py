"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner2e62.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_62."""

import math

from morie.fn import _array_core as np
from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner2e62 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_62,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e62_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_62(
        p_a=0.4, p_z_given_a=0.95, p_z_given_not_a=0.1
    )
    assert isinstance(result, dict)
    assert "posterior" in result
    value = result["posterior"]
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e62_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_62(
        p_a=0.5, p_z_given_a=0.8, p_z_given_not_a=0.2
    )
    assert isinstance(result, dict)
    assert "posterior" in result
    value = result["posterior"]
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0
