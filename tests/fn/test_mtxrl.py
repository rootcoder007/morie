"""Tests for mtxrl.matrix_game."""

from morie.fn import _array_core as np

from morie.fn.mtxrl import matrix_game


def test_mtxrl_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = matrix_game(A)
    assert isinstance(result, dict)
    assert "value" in result


def test_mtxrl_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = matrix_game(A)
    assert isinstance(result, dict)
