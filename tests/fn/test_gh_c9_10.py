"""Tests for gh_c9_10.ghosal_spline_crt."""

from morie.fn import _array_core as np

from morie.fn.gh_c9_10 import ghosal_spline_crt


def test_gh_c9_10_basic():
    """Test basic functionality."""
    result = ghosal_spline_crt()
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_gh_c9_10_edge():
    """Test edge cases."""
    result = ghosal_spline_crt()
    assert isinstance(result, dict)
