"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e4.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_4."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e4 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_4,
)


def _gaussian_density(x):
    return math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e4_basic():
    """Test basic functionality."""
    grid = np.linspace(-5.0, 5.0, 201)
    density = [_gaussian_density(x) for x in grid]
    center = 0.0
    width = 2.0
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_4(
        grid, density, center, width
    )
    assert isinstance(result, dict)
    assert "probability" in result
    prob = result["probability"]
    assert math.isfinite(prob)
    assert 0.0 <= prob <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e4_edge():
    """Test edge cases."""
    grid = np.linspace(-5.0, 5.0, 201)
    density = [_gaussian_density(x) for x in grid]
    center = 1.0
    width = 0.5
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_4(
        grid, density, center, width
    )
    assert isinstance(result, dict)
    assert "probability" in result
    prob = result["probability"]
    assert math.isfinite(prob)
    assert 0.0 <= prob <= 1.0
