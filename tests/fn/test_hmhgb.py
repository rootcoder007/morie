"""Tests for hmhgb.geron_histogram_gradient_boosting."""

from morie.fn import _array_core as np

from morie.fn.hmhgb import geron_histogram_gradient_boosting


def test_hmhgb_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_histogram_gradient_boosting(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "prediction" in result


def test_hmhgb_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_histogram_gradient_boosting(X, y)
    assert isinstance(result, dict)
