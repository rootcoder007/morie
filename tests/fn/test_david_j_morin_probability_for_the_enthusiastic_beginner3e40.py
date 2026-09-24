"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e40.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_40."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e40 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_40,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e40_basic():
    """Test basic functionality."""
    values = np.array([0, 1, 2, 3, 4, 5])
    probs = np.array([0.1, 0.15, 0.2, 0.25, 0.2, 0.1])
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_40(values, probs)
    assert isinstance(result, dict)
    assert "mean" in result
    assert "variance" in result
    mean = result["mean"]
    var = result["variance"]
    assert isinstance(mean, (int, float))
    assert isinstance(var, (int, float))
    assert math.isfinite(mean)
    assert math.isfinite(var)
    assert var >= 0


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e40_edge():
    """Test edge cases."""
    values = np.array([-1.0, 0.0, 1.0])
    probs = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_40(values, probs)
    assert isinstance(result, dict)
    assert "mean" in result
    assert "variance" in result
    mean = result["mean"]
    var = result["variance"]
    assert isinstance(mean, (int, float))
    assert isinstance(var, (int, float))
    assert math.isfinite(mean)
    assert math.isfinite(var)
    assert var >= 0
