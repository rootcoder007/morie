"""Tests for wsmrgr.wasserman_ridge."""

from morie.fn import _array_core as np

from morie.fn.wsmrgr import wasserman_ridge


def test_wsmrgr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    lambda_ = 0.1
    result = wasserman_ridge(X, y, lambda_)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wsmrgr_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    lambda_ = 0.1
    result = wasserman_ridge(X, y, lambda_)
    assert isinstance(result, dict)
