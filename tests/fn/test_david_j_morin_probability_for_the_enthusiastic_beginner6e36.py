"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e36.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_36."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e36 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_36,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e36_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    r = float(rng.uniform(-0.8, 0.8))
    sigma_x = float(rng.uniform(0.5, 3.0))
    sigma_y = float(rng.uniform(0.5, 3.0))
    expected = r * sigma_x / sigma_y
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_36(r, sigma_x, sigma_y)
    assert hasattr(result, "slope")
    assert math.isclose(result.slope, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e36_edge():
    """Test edge cases."""
    r, sigma_x, sigma_y = 0.5, 1.0, 1.0
    expected = r * sigma_x / sigma_y
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_36(r, sigma_x, sigma_y)
    assert hasattr(result, "slope")
    assert math.isclose(result.slope, expected, rel_tol=1e-12, abs_tol=1e-12)
