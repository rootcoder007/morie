"""Tests for gh_c10_10.ghosal_frs_poireg."""

from morie.fn import _array_core as np

from morie.fn.gh_c10_10 import ghosal_frs_poireg


def test_gh_c10_10_basic():
    """Test basic functionality."""
    result = ghosal_frs_poireg()
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_gh_c10_10_edge():
    """Test edge cases."""
    result = ghosal_frs_poireg()
    assert isinstance(result, dict)
