"""Tests for agmuef.muzero_efficient_exploration."""

from morie.fn import _array_core as np

from morie.fn.agmuef import muzero_efficient_exploration


def test_agmuef_basic():
    """Test basic functionality."""
    Q = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    N = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    P = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = muzero_efficient_exploration(Q, N, P)
    assert isinstance(result, dict)
    assert "score" in result
def test_agmuef_edge():
    """Test edge cases."""
    Q = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    N = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    P = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = muzero_efficient_exploration(Q, N, P)
    assert isinstance(result, dict)
