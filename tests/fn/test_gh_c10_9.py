"""Tests for gh_c10_9.ghosal_frs_binreg."""

from morie.fn import _array_core as np

from morie.fn.gh_c10_9 import ghosal_frs_binreg


def test_gh_c10_9_basic():
    """Test basic functionality."""
    result = ghosal_frs_binreg()
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_gh_c10_9_edge():
    """Test edge cases."""
    result = ghosal_frs_binreg()
    assert isinstance(result, dict)
