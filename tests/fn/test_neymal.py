"""Tests for neymal.neyman_allocation."""

from morie.fn import _array_core as np

from morie.fn.neymal import neyman_allocation


def test_neymal_basic():
    """Test basic functionality."""
    N = 0.5
    Nh = 0.5
    Sh = 0.5
    n = 0.5
    result = neyman_allocation(N, Nh, Sh, n)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_neymal_edge():
    """Test edge cases."""
    N = 0.5
    Nh = 0.5
    Sh = 0.5
    n = 0.5
    result = neyman_allocation(N, Nh, Sh, n)
    assert isinstance(result, dict)
