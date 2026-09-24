"""Tests for noncentrality_delta_r.noncentrality_delta_r."""

from morie.fn import _array_core as np

from morie.fn.noncentrality_delta_r import noncentrality_delta_r


def test_ca8e6_basic():
    """Test basic functionality."""
    r = 0.5
    n = 5
    result = noncentrality_delta_r(r, n)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca8e6_edge():
    """Test edge cases."""
    r = 0.5
    n = 5
    result = noncentrality_delta_r(r, n)
    assert isinstance(result, dict)
