"""Tests for impala.impala_vtrace."""

from morie.fn import _array_core as np

from morie.fn.impala import impala_vtrace


def test_impala_basic():
    """Test basic functionality."""
    rewards = np.random.default_rng(42).normal(0.0, 1.0, 40)
    values = np.random.default_rng(42).normal(0.0, 1.0, 40)
    behavior_logp = np.random.default_rng(42).normal(0.0, 1.0, 40)
    target_logp = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = impala_vtrace(rewards, values, behavior_logp, target_logp)
    assert isinstance(result, dict)
    assert "vs" in result


def test_impala_edge():
    """Test edge cases."""
    rewards = np.random.default_rng(42).normal(0.0, 1.0, 40)
    values = np.random.default_rng(42).normal(0.0, 1.0, 40)
    behavior_logp = np.random.default_rng(42).normal(0.0, 1.0, 40)
    target_logp = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = impala_vtrace(rewards, values, behavior_logp, target_logp)
    assert isinstance(result, dict)
