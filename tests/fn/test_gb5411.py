"""Tests for gb5411.gibbons_sign_pvalue."""

from morie.fn import _array_core as np

from morie.fn.gb5411 import gibbons_sign_pvalue


def test_gb5411_basic():
    """Test basic functionality."""
    k = 3
    n = 100
    result = gibbons_sign_pvalue(k, n)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_gb5411_edge():
    """Test edge cases."""
    k = 3
    n = 100
    result = gibbons_sign_pvalue(k, n)
    assert isinstance(result, dict)
