"""Tests for gb_kws.gibbons_kw_chi2_approx."""

from morie.fn import _array_core as np

from morie.fn.gb_kws import gibbons_kw_chi2_approx


def test_gb_kws_basic():
    """Test basic functionality."""
    h = 3
    k = 5
    result = gibbons_kw_chi2_approx(h, k)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_gb_kws_edge():
    """Test edge cases."""
    h = 3
    k = 5
    result = gibbons_kw_chi2_approx(h, k)
    assert isinstance(result, dict)
