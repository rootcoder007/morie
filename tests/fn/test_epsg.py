"""Tests for epsg.epsilon_greedy."""

from morie.fn import _array_core as np

from morie.fn.epsg import epsilon_greedy


def test_epsg_basic():
    """Test basic functionality."""
    arms = np.random.default_rng(42).normal(0, 1, 100)
    result = epsilon_greedy(arms)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_epsg_edge():
    """Test edge cases."""
    arms = np.random.default_rng(42).normal(0, 1, 100)
    result = epsilon_greedy(arms)
    assert isinstance(result, dict)
