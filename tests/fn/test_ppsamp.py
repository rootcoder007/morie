"""Tests for ppsamp.pps_sampling."""

from morie.fn import _array_core as np

from morie.fn.ppsamp import pps_sampling


def test_ppsamp_basic():
    """Test basic functionality."""
    y = 0.5
    size = 0.5
    n = 1
    result = pps_sampling(y, size, n)
    assert isinstance(result, dict)
    assert "estimate" in result or "pi" in result


def test_ppsamp_edge():
    """Test edge cases."""
    y = 0.5
    size = 0.5
    n = 1
    result = pps_sampling(y, size, n)
    assert isinstance(result, dict)
