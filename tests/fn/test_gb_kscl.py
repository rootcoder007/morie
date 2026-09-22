"""Tests for gb_kscl.gibbons_ks_critical_values."""

from morie.fn.gb_kscl import gibbons_ks_critical_values


def test_gb_kscl_basic():
    """Test basic functionality."""
    n = 100
    alpha = 0.05
    result = gibbons_ks_critical_values(n, alpha)
    assert isinstance(result, dict)
    assert "dcrit" in result
def test_gb_kscl_edge():
    """Test edge cases."""
    n = 100
    alpha = 0.05
    result = gibbons_ks_critical_values(n, alpha)
    assert isinstance(result, dict)
