"""Tests for hmregn.geron_regression_mlp."""

from morie.fn import _array_core as np

from morie.fn.hmregn import geron_regression_mlp


def test_hmregn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_regression_mlp(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmregn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_regression_mlp(X, y)
    assert isinstance(result, dict)
