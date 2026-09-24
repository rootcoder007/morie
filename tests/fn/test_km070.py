"""Tests for km070.kamath_ch5_rlhf_optimal_policy."""

from morie.fn import _array_core as np

from morie.fn.km070 import kamath_ch5_rlhf_optimal_policy


def test_km070_basic():
    """Test basic functionality."""
    pi_ref = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    r = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    beta = 0.1
    result = kamath_ch5_rlhf_optimal_policy(pi_ref, r, beta)
    assert isinstance(result, dict)
    assert "estimate" in result or "pi" in result


def test_km070_edge():
    """Test edge cases."""
    pi_ref = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    r = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    beta = 0.1
    result = kamath_ch5_rlhf_optimal_policy(pi_ref, r, beta)
    assert isinstance(result, dict)
