"""Tests for mctsel.mcts_selection."""

from morie.fn import _array_core as np

from morie.fn.mctsel import mcts_selection


def test_mctsel_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0.0, 1.0, 40)
    N = np.random.default_rng(42).normal(0.0, 1.0, 40)
    P = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mcts_selection(Q, N, P)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_mctsel_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0.0, 1.0, 40)
    N = np.random.default_rng(42).normal(0.0, 1.0, 40)
    P = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mcts_selection(Q, N, P)
    assert isinstance(result, dict)
