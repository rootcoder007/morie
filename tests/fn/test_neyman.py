"""Tests for neyman.neyman_allocation."""

from morie.fn import _array_core as np

from morie.fn.neyman import neyman_allocation


def test_neyman_basic():
    """Test basic functionality."""
    y = 0.5
    N_h = 0.5
    S_h = 0.5
    n = 0.5
    result = neyman_allocation(y, N_h, S_h, n)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_neyman_edge():
    """Test edge cases."""
    y = 0.5
    N_h = 0.5
    S_h = 0.5
    n = 0.5
    result = neyman_allocation(y, N_h, S_h, n)
    assert isinstance(result, dict)
