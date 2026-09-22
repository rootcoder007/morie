"""Tests for gh_c8_8.ghosal_gauss_reg_crt."""

from morie.fn import _array_core as np

from morie.fn.gh_c8_8 import ghosal_gauss_reg_crt


def test_gh_c8_8_basic():
    """Test basic functionality."""
    result = ghosal_gauss_reg_crt()
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_gh_c8_8_edge():
    """Test edge cases."""
    result = ghosal_gauss_reg_crt()
    assert isinstance(result, dict)
