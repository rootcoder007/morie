"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e2.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_2."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e2 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_2,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e2_basic():
    """Test basic functionality."""
    grid = np.linspace(-3.0, 3.0, 200)
    density = [
        math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi) for x in grid
    ]
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_2(
        grid, density, -1.0, 1.0
    )
    assert isinstance(result, dict)
    assert "probability" in result
    prob = result["probability"]
    assert math.isfinite(prob)
    assert 0.0 <= prob <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e2_edge():
    """Test edge cases."""
    grid = np.linspace(-3.0, 3.0, 200)
    density = [
        math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi) for x in grid
    ]
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_2(
        grid, density, 0.0, 0.0
    )
    assert isinstance(result, dict)
    assert "probability" in result
    prob = result["probability"]
    assert math.isfinite(prob)
    assert 0.0 <= prob <= 1.0
