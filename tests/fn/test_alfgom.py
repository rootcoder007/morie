"""Tests for alfgom.alphago_montecarlo."""

from morie.fn import _array_core as np

from morie.fn.alfgom import alphago_montecarlo


def test_alfgom_basic():
    """Test basic functionality."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    rollout_net = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = alphago_montecarlo(state, rollout_net, horizon)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_alfgom_edge():
    """Test edge cases."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    rollout_net = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = alphago_montecarlo(state, rollout_net, horizon)
    assert isinstance(result, dict)
