"""Tests for gh_c7_10.ghosal_mono_reg_con."""

from morie.fn import _array_core as np

from morie.fn.gh_c7_10 import ghosal_mono_reg_con


def test_gh_c7_10_basic():
    """Test basic functionality."""
    result = ghosal_mono_reg_con()
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_gh_c7_10_edge():
    """Test edge cases."""
    result = ghosal_mono_reg_con()
    assert isinstance(result, dict)
