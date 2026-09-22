"""Tests for gb1122.gibbons_kendall_null."""

from morie.fn.gb1122 import gibbons_kendall_null


def test_gb1122_basic():
    """Test basic functionality."""
    n = 100
    result = gibbons_kendall_null(n)
    assert isinstance(result, dict)
    assert "support" in result
def test_gb1122_edge():
    """Test edge cases."""
    n = 100
    result = gibbons_kendall_null(n)
    assert isinstance(result, dict)
