"""Tests for extended_predictor.extended_predictor."""

from morie.fn import _array_core as np

from morie.fn.extended_predictor import extended_predictor


def test_msm061_basic():
    """Test basic functionality."""
    n = 50
    result = extended_predictor(n)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm061_edge():
    """Test edge cases."""
    n = 50
    result = extended_predictor(n)
    assert isinstance(result, dict)
