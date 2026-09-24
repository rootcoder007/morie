"""Tests for hml1c.geron_one_cycle."""

from morie.fn import _array_core as np

from morie.fn.hml1c import geron_one_cycle


def test_hml1c_basic():
    """Test basic functionality."""
    t = 0.5
    T = 5
    lr_max = 5
    lr_min = 0.5
    result = geron_one_cycle(t, T, lr_max, lr_min)
    assert isinstance(result, dict)
    assert "estimate" in result or "lr" in result


def test_hml1c_edge():
    """Test edge cases."""
    t = 0.5
    T = 5
    lr_max = 5
    lr_min = 0.5
    result = geron_one_cycle(t, T, lr_max, lr_min)
    assert isinstance(result, dict)
