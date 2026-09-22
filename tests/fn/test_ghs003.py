"""Tests for ghs003.ghosal_ch2_basis_truncation_error."""

from morie.fn import _array_core as np

from morie.fn.ghs003 import ghosal_ch2_basis_truncation_error


def test_ghs003_basic():
    """Test basic functionality."""
    result = ghosal_ch2_basis_truncation_error()
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ghs003_edge():
    """Test edge cases."""
    result = ghosal_ch2_basis_truncation_error()
    assert isinstance(result, dict)
