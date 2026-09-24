"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e60.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_60."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e60 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_60,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e60_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    p = 0.5
    k = int(rng.integers(0, n + 1))
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_60(k, n, p)
    assert isinstance(result, dict)
    assert "probability" in result
    prob = result["probability"]
    assert math.isfinite(prob)
    assert 0.0 <= prob <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e60_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_60(0, 10, 0.5)
    assert isinstance(result, dict)
    assert "probability" in result
    prob = result["probability"]
    assert math.isfinite(prob)
    assert 0.0 <= prob <= 1.0
