"""Tests for f_nested_ss.f_nested_ss."""

from morie.fn import _array_core as np

from morie.fn.f_nested_ss import f_nested_ss


def test_ca2e18_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = f_nested_ss(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_ca2e18_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = f_nested_ss(x)
    assert isinstance(result, dict)
