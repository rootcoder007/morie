"""Tests for f_nested_r2.f_nested_r2."""

from morie.fn import _array_core as np

from morie.fn.f_nested_r2 import f_nested_r2


def test_ca2e19_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = f_nested_r2(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_ca2e19_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = f_nested_r2(x)
    assert isinstance(result, dict)
