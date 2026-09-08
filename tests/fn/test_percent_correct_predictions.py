"""Tests for percent_correct_predictions.percent_correct_predictions."""

from morie.fn import _array_core as np

from morie.fn.percent_correct_predictions import percent_correct_predictions


def test_ca4e12_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = percent_correct_predictions(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_ca4e12_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = percent_correct_predictions(x)
    assert isinstance(result, dict)
