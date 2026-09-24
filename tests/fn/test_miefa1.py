"""Tests for miefa1.mi_fmi."""

from morie.fn import _array_core as np

from morie.fn.miefa1 import mi_fmi


def test_miefa1_basic():
    """Test basic functionality."""
    between = 0.1
    within = 0.1
    m = 5
    result = mi_fmi(between, within, m)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_miefa1_edge():
    """Test edge cases."""
    between = 0.1
    within = 0.1
    m = 5
    result = mi_fmi(between, within, m)
    assert isinstance(result, dict)
