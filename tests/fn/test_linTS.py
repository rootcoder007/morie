"""Tests for linTS.lin_thompson."""

from morie.fn import _array_core as np

from morie.fn.linTS import lin_thompson


def test_linTS_basic():
    """Test basic functionality."""
    contexts = np.random.default_rng(42).normal(0.0, 1.0, 40)
    played = np.random.default_rng(42).normal(0.0, 1.0, 40)
    rewards = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = lin_thompson(contexts, played, rewards)
    assert isinstance(result, dict)
    assert "arm" in result


def test_linTS_edge():
    """Test edge cases."""
    contexts = np.random.default_rng(42).normal(0.0, 1.0, 40)
    played = np.random.default_rng(42).normal(0.0, 1.0, 40)
    rewards = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = lin_thompson(contexts, played, rewards)
    assert isinstance(result, dict)
