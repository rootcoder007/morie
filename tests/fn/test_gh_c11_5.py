"""Tests for gh_c11_5.ghosal_gp_binreg_crt."""

from morie.fn import _array_core as np

from morie.fn.gh_c11_5 import ghosal_gp_binreg_crt


def test_gh_c11_5_basic():
    """Test basic functionality."""
    result = ghosal_gp_binreg_crt()
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_gh_c11_5_edge():
    """Test edge cases."""
    result = ghosal_gp_binreg_crt()
    assert isinstance(result, dict)
