"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e83.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_83."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e83 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_83,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e83_basic():
    """Test basic functionality."""
    tau = 1.5
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_83(tau)
    assert isinstance(result, dict)
    assert "mean" in result
    assert math.isfinite(result["mean"])


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e83_edge():
    """Test edge cases."""
    tau = 0.5
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_83(tau)
    assert isinstance(result, dict)
    assert "mean" in result
    assert math.isfinite(result["mean"])
