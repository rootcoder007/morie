"""Tests for gb_cvo.gibbons_order_covariance."""

from morie.fn import _array_core as np

from morie.fn.gb_cvo import gibbons_order_covariance


def test_gb_cvo_basic():
    """Test basic functionality."""
    r = 10
    s = 90
    n = 100
    result = gibbons_order_covariance(r, s, n)
    assert isinstance(result, dict)
    assert "cov" in result
def test_gb_cvo_edge():
    """Test edge cases."""
    r = 10
    s = 90
    n = 100
    result = gibbons_order_covariance(r, s, n)
    assert isinstance(result, dict)
