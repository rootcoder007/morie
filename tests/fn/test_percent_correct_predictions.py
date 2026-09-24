"""Tests for percent_correct_predictions.percent_correct_predictions."""

from morie.fn import _array_core as np

from morie.fn.percent_correct_predictions import percent_correct_predictions


def test_ca4e12_basic():
    """Test basic functionality."""
    n_correct = 0.5
    n_total = 0.5
    result = percent_correct_predictions(n_correct, n_total)
    assert isinstance(result, dict)
    assert "value" in result or "value" in result


def test_ca4e12_edge():
    """Test edge cases."""
    n_correct = 0.5
    n_total = 0.5
    result = percent_correct_predictions(n_correct, n_total)
    assert isinstance(result, dict)
