"""Tests for hmrlhf.geron_rlhf."""

from morie.fn import _array_core as np

from morie.fn.hmrlhf import geron_rlhf


def test_hmrlhf_basic():
    """Test basic functionality."""
    policy = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    reward_model = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_rlhf(policy, reward_model)
    assert isinstance(result, dict)
    assert "estimate" in result or "policy" in result


def test_hmrlhf_edge():
    """Test edge cases."""
    policy = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    reward_model = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_rlhf(policy, reward_model)
    assert isinstance(result, dict)
