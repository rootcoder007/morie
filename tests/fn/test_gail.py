"""Tests for gail.gail."""

from morie.fn import _array_core as np

from morie.fn.gail import gail


def test_gail_basic():
    """Test basic functionality."""
    expert_states = np.random.default_rng(42).normal(0, 1, 100)
    expert_actions = np.random.default_rng(42).normal(0, 1, 100)
    policy_states = np.random.default_rng(42).normal(0, 1, 100)
    policy_actions = np.random.default_rng(42).normal(0, 1, 100)
    result = gail(expert_states, expert_actions, policy_states, policy_actions)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_gail_edge():
    """Test edge cases."""
    expert_states = np.random.default_rng(42).normal(0, 1, 100)
    expert_actions = np.random.default_rng(42).normal(0, 1, 100)
    policy_states = np.random.default_rng(42).normal(0, 1, 100)
    policy_actions = np.random.default_rng(42).normal(0, 1, 100)
    result = gail(expert_states, expert_actions, policy_states, policy_actions)
    assert isinstance(result, dict)
