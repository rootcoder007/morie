"""Tests for hmsgdc.geron_sgd_classifier."""

from morie.fn import _array_core as np

from morie.fn.hmsgdc import geron_sgd_classifier


def test_hmsgdc_basic():
    """Test basic functionality."""
    X = [[2.0, -1.0]]
    y = [1]
    result = geron_sgd_classifier(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmsgdc_edge():
    """Test edge cases."""
    X = [[2.0, -1.0]]
    y = [1]
    result = geron_sgd_classifier(X, y)
    assert isinstance(result, dict)
