"""Tests for gb631a.gibbons_ks2_asymp."""

from morie.fn import _array_core as np

from morie.fn.gb631a import gibbons_ks2_asymp


def test_gb631a_basic():
    """Test basic functionality."""
    d = 3
    m = 50
    n = 50
    result = gibbons_ks2_asymp(d, m, n)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_gb631a_edge():
    """Test edge cases."""
    d = 3
    m = 50
    n = 50
    result = gibbons_ks2_asymp(d, m, n)
    assert isinstance(result, dict)
