"""Tests for gh_c10_8.ghosal_frs_reg."""

from morie.fn import _array_core as np

from morie.fn.gh_c10_8 import ghosal_frs_reg


def test_gh_c10_8_basic():
    """Test basic functionality."""
    result = ghosal_frs_reg()
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_gh_c10_8_edge():
    """Test edge cases."""
    result = ghosal_frs_reg()
    assert isinstance(result, dict)
