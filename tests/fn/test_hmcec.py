"""Tests for hmcec.geron_cross_entropy_cost."""

from morie.fn import _array_core as np

from morie.fn.hmcec import geron_cross_entropy_cost


def test_hmcec_basic():
    """Test basic functionality."""
    X = 5
    Y = 5
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_cross_entropy_cost(X, Y, theta)
    assert isinstance(result, dict)
    assert "estimate" in result or "cost" in result


def test_hmcec_edge():
    """Test edge cases."""
    X = 5
    Y = 5
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_cross_entropy_cost(X, Y, theta)
    assert isinstance(result, dict)
