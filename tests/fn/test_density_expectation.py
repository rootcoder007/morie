"""Tests for density_expectation.density_expectation."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.density_expectation import (
    density_expectation,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e55_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    grid = np.linspace(-5.0, 5.0, 201)
    density = [math.exp(-0.5 * (x ** 2)) / math.sqrt(2 * math.pi) for x in grid]
    result = density_expectation(grid, density)
    assert isinstance(result, dict)
    assert "expectation" in result
    assert math.isfinite(result["expectation"])
    # Standard normal density is symmetric about 0, so E[X] = 0
    assert abs(result["expectation"]) < 0.01


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e55_edge():
    """Test edge cases."""
    # Uniform density rho(x) = 1 on the interval [0, 1]
    n = 51
    grid = np.linspace(0.0, 1.0, n)
    density = [1.0] * n
    result = density_expectation(grid, density)
    assert isinstance(result, dict)
    assert "expectation" in result
    assert math.isfinite(result["expectation"])
    # E[X] for uniform on [0, 1] is 0.5
    assert abs(result["expectation"] - 0.5) < 0.01
