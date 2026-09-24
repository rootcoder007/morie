"""Tests for propal.proportional_allocation."""

from morie.fn import _array_core as np

from morie.fn.propal import proportional_allocation


def test_propal_basic():
    """Test basic functionality."""
    N = 0.5
    Nh = 0.5
    n = 0.5
    result = proportional_allocation(N, Nh, n)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_propal_edge():
    """Test edge cases."""
    N = 0.5
    Nh = 0.5
    n = 0.5
    result = proportional_allocation(N, Nh, n)
    assert isinstance(result, dict)
