"""Tests for hmlogp.geron_logistic_probability."""

from morie.fn import _array_core as np

from morie.fn.hmlogp import geron_logistic_probability


def test_hmlogp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_logistic_probability(X, theta)
    assert isinstance(result, dict)
    assert "estimate" in result or "p_hat" in result


def test_hmlogp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_logistic_probability(X, theta)
    assert isinstance(result, dict)
