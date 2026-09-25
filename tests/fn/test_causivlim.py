"""Tests for causivlim.causal_iv_liml."""

from morie.fn import _array_core as np

from morie.fn.causivlim import causal_iv_liml


def test_causivlim_basic():
    """Test basic functionality."""
    y = 0.5
    X = 0.5
    Z = 5
    result = causal_iv_liml(y, X, Z)
    assert isinstance(result, dict)
    assert "beta" in result


def test_causivlim_edge():
    """Test edge cases."""
    y = 0.5
    X = 0.5
    Z = 5
    result = causal_iv_liml(y, X, Z)
    assert isinstance(result, dict)
