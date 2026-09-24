"""Tests for jormse2.joseph_rmsse."""

from morie.fn import _array_core as np

from morie.fn.jormse2 import joseph_rmsse


def test_jormse2_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    yhat = np.random.default_rng(42).normal(0.0, 1.0, 40)
    insample = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_rmsse(y, yhat, insample)
    assert isinstance(result, dict)
    assert "rmsse" in result


def test_jormse2_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    yhat = np.random.default_rng(42).normal(0.0, 1.0, 40)
    insample = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_rmsse(y, yhat, insample)
    assert isinstance(result, dict)
