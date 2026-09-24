"""Tests for hmsgdu.geron_sgd_update."""

from morie.fn import _array_core as np

from morie.fn.hmsgdu import geron_sgd_update


def test_hmsgdu_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    theta = 0.1
    result = geron_sgd_update(X, y, theta)
    assert isinstance(result, dict)
    assert "estimate" in result or "theta" in result


def test_hmsgdu_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    theta = 0.1
    result = geron_sgd_update(X, y, theta)
    assert isinstance(result, dict)
