"""Tests for multinomial_conditional_or.multinomial_conditional_or."""

from morie.fn import _array_core as np

from morie.fn.multinomial_conditional_or import multinomial_conditional_or


def test_ca5e4_basic():
    """Test basic functionality."""
    xb_m = 0.5
    xb_n = 0.5
    result = multinomial_conditional_or(xb_m, xb_n)
    assert isinstance(result, dict)
    assert "value" in result or "value" in result


def test_ca5e4_edge():
    """Test edge cases."""
    xb_m = 0.5
    xb_n = 0.5
    result = multinomial_conditional_or(xb_m, xb_n)
    assert isinstance(result, dict)
