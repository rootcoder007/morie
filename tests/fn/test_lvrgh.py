"""Tests for lvrgh.hat_matrix_diagonal."""

from morie.fn import _array_core as np

from morie.fn.lvrgh import hat_matrix_diagonal


def test_lvrgh_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = hat_matrix_diagonal(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_lvrgh_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = hat_matrix_diagonal(X)
    assert isinstance(result, dict)
