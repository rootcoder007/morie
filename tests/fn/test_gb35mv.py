"""Tests for gb35mv.gibbons_rvn_moments."""

from morie.fn.gb35mv import gibbons_rvn_moments


def test_gb35mv_basic():
    """Test basic functionality."""
    n = 100
    result = gibbons_rvn_moments(n)
    assert isinstance(result, dict)
    assert "mean" in result
def test_gb35mv_edge():
    """Test edge cases."""
    n = 100
    result = gibbons_rvn_moments(n)
    assert isinstance(result, dict)
