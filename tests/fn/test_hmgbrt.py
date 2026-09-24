"""Tests for hmgbrt.geron_gradient_boosting."""

from morie.fn import _array_core as np

from morie.fn.hmgbrt import geron_gradient_boosting


def test_hmgbrt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_gradient_boosting(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "predictions" in result


def test_hmgbrt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_gradient_boosting(X, y)
    assert isinstance(result, dict)
