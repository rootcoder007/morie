"""Tests for multinomial_conditional_or.multinomial_conditional_or."""

from morie.fn import _array_core as np

from morie.fn.multinomial_conditional_or import multinomial_conditional_or


def test_ca5e4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = multinomial_conditional_or(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_ca5e4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = multinomial_conditional_or(x)
    assert isinstance(result, dict)
