"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e66.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_66."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e66 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_66,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e66_basic():
    """Test basic functionality."""
    grid_x = np.linspace(-3.0, 3.0, 61)
    raw_x = [math.exp(-x * x / 2.0) for x in grid_x]
    total_x = sum(raw_x)
    density_x = [r / total_x for r in raw_x]

    grid_y = np.linspace(-3.0, 3.0, 61)
    raw_y = [math.exp(-y * y / 2.0) for y in grid_y]
    total_y = sum(raw_y)
    density_y = [r / total_y for r in raw_y]

    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_66(
        grid_x, density_x, grid_y, density_y, 0.0, 0.2
    )
    assert isinstance(result, dict)
    assert 'probability' in result
    prob = float(result['probability'])
    assert math.isfinite(prob)
    assert 0.0 <= prob <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e66_edge():
    """Test edge cases."""
    grid_x = np.linspace(-5.0, 5.0, 101)
    total_x = float(len(grid_x))
    density_x = [1.0 / total_x for _ in grid_x]

    grid_y = np.linspace(-5.0, 5.0, 101)
    total_y = float(len(grid_y))
    density_y = [1.0 / total_y for _ in grid_y]

    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_66(
        grid_x, density_x, grid_y, density_y, 1.0, 0.5
    )
    assert isinstance(result, dict)
    assert 'probability' in result
    prob = float(result['probability'])
    assert math.isfinite(prob)
    assert 0.0 <= prob <= 1.0
