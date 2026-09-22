"""Tests for ghs006.ghosal_ch2_feller_density_approximation."""

from morie.fn import _array_core as np

from morie.fn.ghs006 import ghosal_ch2_feller_density_approximation


def test_ghs006_basic():
    """Test basic functionality."""
    result = ghosal_ch2_feller_density_approximation()
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ghs006_edge():
    """Test edge cases."""
    result = ghosal_ch2_feller_density_approximation()
    assert isinstance(result, dict)
