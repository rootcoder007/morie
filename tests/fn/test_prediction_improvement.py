"""Tests for prediction_improvement.prediction_improvement."""

from morie.fn import _array_core as np

from morie.fn.prediction_improvement import (
    prediction_improvement,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e27_basic():
    """Test basic functionality."""
    r = 0.5
    result = prediction_improvement(r)
    assert isinstance(result, dict)
    assert "r" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e27_edge():
    """Test edge cases."""
    r = 0.5
    result = prediction_improvement(r)
    assert isinstance(result, dict)
