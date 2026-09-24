"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e25.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_25."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e25 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_25,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e25_basic():
    """Test basic functionality."""
    t = 1.0
    dt = 0.1
    lam = 1.0
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_25(
        t, dt, lam
    )
    assert isinstance(result, dict)
    assert "probability" in result
    p = result["probability"]
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e25_edge():
    """Test edge cases."""
    t = 0.5
    dt = 0.01
    lam = 2.0
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_25(
        t, dt, lam
    )
    assert isinstance(result, dict)
    assert "probability" in result
    p = result["probability"]
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0
