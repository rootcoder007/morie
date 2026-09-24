"""Tests for hmhebb.geron_hebb_rule."""

from morie.fn import _array_core as np

from morie.fn.hmhebb import geron_hebb_rule


def test_hmhebb_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_hebb_rule(X, Y)
    assert isinstance(result, dict)
    assert "estimate" in result or "dW" in result


def test_hmhebb_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_hebb_rule(X, Y)
    assert isinstance(result, dict)
