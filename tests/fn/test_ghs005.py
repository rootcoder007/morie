"""Tests for ghs005.ghosal_ch2_location_scale_mixture_limit."""

from morie.fn import _array_core as np

from morie.fn.ghs005 import ghosal_ch2_location_scale_mixture_limit


def test_ghs005_basic():
    """Test basic functionality."""
    result = ghosal_ch2_location_scale_mixture_limit()
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ghs005_edge():
    """Test edge cases."""
    result = ghosal_ch2_location_scale_mixture_limit()
    assert isinstance(result, dict)
