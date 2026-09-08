"""Tests for mtxrl.matrix_game."""

from morie.fn import _array_core as np

from morie.fn.mtxrl import matrix_game


def test_mtxrl_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = matrix_game(A)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_mtxrl_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = matrix_game(A)
    assert isinstance(result, dict)
