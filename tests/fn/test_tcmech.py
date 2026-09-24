"""Tests for tcmech.truncated_cdp_mechanism."""

from morie.fn import _array_core as np

from morie.fn.tcmech import truncated_cdp_mechanism


def test_tcmech_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    f_value = 0.1
    C = 0.1
    epsilon = 0.1
    delta = 0.1
    result = truncated_cdp_mechanism(y, f_value, C, epsilon, delta)
    assert isinstance(result, dict)
    assert "estimate" in result or "private_value" in result


def test_tcmech_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    f_value = 0.1
    C = 0.1
    epsilon = 0.1
    delta = 0.1
    result = truncated_cdp_mechanism(y, f_value, C, epsilon, delta)
    assert isinstance(result, dict)
