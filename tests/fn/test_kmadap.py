"""Tests for kmadap.kamath_houlsby_adapter."""

from morie.fn import _array_core as np

from morie.fn.kmadap import kamath_houlsby_adapter


def test_kmadap_basic():
    """Test basic functionality."""
    h = [1.0, 2.0, 3.0]
    W_down = [[0.0, 0.0, 0.0]]
    W_up = [[0.0], [0.0], [0.0]]
    result = kamath_houlsby_adapter(h, W_down, W_up)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmadap_edge():
    """Test edge cases."""
    h = [1.0, 2.0, 3.0]
    W_down = [[0.0, 0.0, 0.0]]
    W_up = [[0.0], [0.0], [0.0]]
    result = kamath_houlsby_adapter(h, W_down, W_up)
    assert isinstance(result, dict)
