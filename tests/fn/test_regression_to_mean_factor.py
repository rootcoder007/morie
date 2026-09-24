"""Tests for regression_to_mean_factor.regression_to_mean_factor."""

from morie.fn import _array_core as np

from morie.fn.regression_to_mean_factor import (
    regression_to_mean_factor,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e40_basic():
    """Test basic functionality."""
    r = 0.5
    y1 = 0.5
    result = regression_to_mean_factor(r, y1)
    assert isinstance(result, dict)
    assert "yavg" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e40_edge():
    """Test edge cases."""
    r = 0.5
    y1 = 0.5
    result = regression_to_mean_factor(r, y1)
    assert isinstance(result, dict)
