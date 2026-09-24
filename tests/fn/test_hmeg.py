"""Tests for hmeg.geron_epsilon_greedy."""

from morie.fn import _array_core as np

from morie.fn.hmeg import geron_epsilon_greedy


def test_hmeg_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    s = 5
    epsilon = 0.1
    result = geron_epsilon_greedy(Q, s, epsilon)
    assert isinstance(result, dict)
    assert "estimate" in result or "action" in result


def test_hmeg_edge():
    """Test edge cases."""
    Q = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    s = 5
    epsilon = 0.1
    result = geron_epsilon_greedy(Q, s, epsilon)
    assert isinstance(result, dict)
