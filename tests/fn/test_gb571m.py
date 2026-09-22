"""Tests for gb571m.gibbons_wsrt_mean."""

from morie.fn.gb571m import gibbons_wsrt_mean


def test_gb571m_basic():
    """Test basic functionality."""
    n = 100
    result = gibbons_wsrt_mean(n)
    assert isinstance(result, dict)
    assert "mean" in result
def test_gb571m_edge():
    """Test edge cases."""
    n = 100
    result = gibbons_wsrt_mean(n)
    assert isinstance(result, dict)
