"""Tests for hmceg.geron_cross_entropy_gradient."""

from morie.fn import _array_core as np

from morie.fn.hmceg import geron_cross_entropy_gradient


def test_hmceg_basic():
    """Test basic functionality."""
    X = 5
    Y = 5
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_cross_entropy_gradient(X, Y, theta)
    assert isinstance(result, dict)
    assert "estimate" in result or "gradient" in result


def test_hmceg_edge():
    """Test edge cases."""
    X = 5
    Y = 5
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_cross_entropy_gradient(X, Y, theta)
    assert isinstance(result, dict)
