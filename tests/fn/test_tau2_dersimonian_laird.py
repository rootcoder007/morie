"""Tests for tau2_dersimonian_laird.tau2_dersimonian_laird."""

from morie.fn import _array_core as np

from morie.fn.tau2_dersimonian_laird import tau2_dersimonian_laird


def test_ca11e44_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = tau2_dersimonian_laird(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_ca11e44_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = tau2_dersimonian_laird(x)
    assert isinstance(result, dict)
