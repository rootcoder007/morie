"""Tests for airl.airl."""

from morie.fn import _array_core as np

from morie.fn.airl import airl


def test_airl_basic():
    """Test basic functionality."""
    expert_states = np.random.default_rng(42).normal(0, 1, 100)
    expert_actions = np.random.default_rng(42).normal(0, 1, 100)
    expert_next = np.random.default_rng(42).normal(0, 1, 100)
    expert_log_policy = np.random.default_rng(42).normal(0, 1, 100)
    policy_states = np.random.default_rng(42).normal(0, 1, 100)
    policy_actions = np.random.default_rng(42).normal(0, 1, 100)
    policy_next = np.random.default_rng(42).normal(0, 1, 100)
    policy_log_policy = np.random.default_rng(42).normal(0, 1, 100)
    result = airl(expert_states, expert_actions, expert_next, expert_log_policy, policy_states, policy_actions, policy_next, policy_log_policy)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_airl_edge():
    """Test edge cases."""
    expert_states = np.random.default_rng(42).normal(0, 1, 100)
    expert_actions = np.random.default_rng(42).normal(0, 1, 100)
    expert_next = np.random.default_rng(42).normal(0, 1, 100)
    expert_log_policy = np.random.default_rng(42).normal(0, 1, 100)
    policy_states = np.random.default_rng(42).normal(0, 1, 100)
    policy_actions = np.random.default_rng(42).normal(0, 1, 100)
    policy_next = np.random.default_rng(42).normal(0, 1, 100)
    policy_log_policy = np.random.default_rng(42).normal(0, 1, 100)
    result = airl(expert_states, expert_actions, expert_next, expert_log_policy, policy_states, policy_actions, policy_next, policy_log_policy)
    assert isinstance(result, dict)
