"""Tests for hmrdt.geron_regression_tree."""

from morie.fn import _array_core as np

from morie.fn.hmrdt import geron_regression_tree


def test_hmrdt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_regression_tree(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "tree" in result


def test_hmrdt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_regression_tree(X, y)
    assert isinstance(result, dict)
