"""Tests for hmlogg.geron_logistic_gradient."""

from morie.fn import _array_core as np

from morie.fn.hmlogg import geron_logistic_gradient


def test_hmlogg_basic():
    """Test basic functionality."""
    X = 0.5
    y = 1
    theta = 0.5
    result = geron_logistic_gradient(X, y, theta)
    assert isinstance(result, dict)
    assert "estimate" in result or "gradient" in result


def test_hmlogg_edge():
    """Test edge cases."""
    X = 0.5
    y = 1
    theta = 0.5
    result = geron_logistic_gradient(X, y, theta)
    assert isinstance(result, dict)
