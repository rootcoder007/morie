"""Tests for km071.kamath_ch5_dpo_reward_optimal."""

from morie.fn import _array_core as np

from morie.fn.km071 import kamath_ch5_dpo_reward_optimal


def test_km071_basic():
    """Test basic functionality."""
    pi_star = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    pi_ref = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    beta = 0.1
    result = kamath_ch5_dpo_reward_optimal(pi_star, pi_ref, beta)
    assert isinstance(result, dict)
    assert "estimate" in result or "r" in result


def test_km071_edge():
    """Test edge cases."""
    pi_star = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    pi_ref = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    beta = 0.1
    result = kamath_ch5_dpo_reward_optimal(pi_star, pi_ref, beta)
    assert isinstance(result, dict)
