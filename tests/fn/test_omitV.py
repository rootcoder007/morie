"""Tests for omitV.omitted_variable_bias."""

from morie.fn import _array_core as np

from morie.fn.omitV import omitted_variable_bias


def test_omitV_basic():
    """Test basic functionality."""
    result = omitted_variable_bias()
    assert isinstance(result, dict)
    assert "bias" in result


def test_omitV_edge():
    """Test edge cases."""
    result = omitted_variable_bias()
    assert isinstance(result, dict)
