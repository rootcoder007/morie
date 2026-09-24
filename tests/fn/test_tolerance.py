"""Tests for tolerance.tolerance."""

from morie.fn import _array_core as np

from morie.fn.tolerance import tolerance


def test_ca3e1_basic():
    """Test basic functionality."""
    r2_x = 0.5
    result = tolerance(r2_x)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca3e1_edge():
    """Test edge cases."""
    r2_x = 0.5
    result = tolerance(r2_x)
    assert isinstance(result, dict)
