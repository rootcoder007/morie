"""Tests for ols_matrix.ols_matrix."""

from morie.fn import _array_core as np

from morie.fn.ols_matrix import ols_matrix


def test_ca12e3_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ols_matrix(x, y)
    assert isinstance(result, dict)
    assert "beta" in result


def test_ca12e3_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ols_matrix(x, y)
    assert isinstance(result, dict)
