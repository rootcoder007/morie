"""Tests for gh_c10_11.ghosal_func_reg."""

from morie.fn import _array_core as np

from morie.fn.gh_c10_11 import ghosal_func_reg


def test_gh_c10_11_basic():
    """Test basic functionality."""
    result = ghosal_func_reg()
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_gh_c10_11_edge():
    """Test edge cases."""
    result = ghosal_func_reg()
    assert isinstance(result, dict)
