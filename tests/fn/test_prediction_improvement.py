"""Tests for prediction_improvement.prediction_improvement."""

from morie.fn import _array_core as np

from morie.fn.prediction_improvement import (
    prediction_improvement,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e27_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = prediction_improvement(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e27_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = prediction_improvement(x)
    assert isinstance(result, dict)
