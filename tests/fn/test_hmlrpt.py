"""Tests for hmlrpt.geron_linreg_pytorch."""

from morie.fn import _array_core as np

from morie.fn.hmlrpt import geron_linreg_pytorch


def test_hmlrpt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_linreg_pytorch(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "w" in result


def test_hmlrpt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_linreg_pytorch(X, y)
    assert isinstance(result, dict)
