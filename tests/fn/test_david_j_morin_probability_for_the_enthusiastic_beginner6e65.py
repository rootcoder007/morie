"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e65.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_65."""

import math
import warnings

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e65 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_65,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e65_basic():
    """Test basic functionality."""
    # Build a grid and a standard-normal density on it for X
    grid_x = np.linspace(-5.0, 5.0, 51)
    inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
    density_x = [math.exp(-x * x / 2.0) * inv_sqrt2pi for x in grid_x]

    # Build a grid and a standard-normal density on it for Y
    grid_y = np.linspace(-5.0, 5.0, 51)
    density_y = [math.exp(-y * y / 2.0) * inv_sqrt2pi for y in grid_y]

    z = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_65(
            grid_x, density_x, grid_y, density_y, z
        )

    assert isinstance(result, dict)
    assert 'z' in result
    assert 'density' in result
    assert result['z'] == z
    assert math.isfinite(result['density'])
    assert result['density'] >= 0.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e65_edge():
    """Test edge cases."""
    # Small uniform-like grid for X
    grid_x = np.linspace(-1.0, 1.0, 11)
    density_x = np.ones(11)

    # Small uniform-like grid for Y
    grid_y = np.linspace(-1.0, 1.0, 11)
    density_y = np.ones(11)

    z = 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_65(
            grid_x, density_x, grid_y, density_y, z
        )

    assert isinstance(result, dict)
    assert 'z' in result
    assert 'density' in result
    assert result['z'] == z
    assert math.isfinite(result['density'])
    assert result['density'] >= 0.0
