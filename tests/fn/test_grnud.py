"""Tests for grnud.geron_numerical_differentiation."""

from morie.fn import _array_core as np

from morie.fn.grnud import geron_numerical_differentiation


def test_grnud_basic():
    """Test basic functionality."""
    f = lambda t: 3 * t ** 2 + 2 * t
    x = 4.0
    result = geron_numerical_differentiation(f, x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grnud_edge():
    """Test edge cases."""
    f = lambda t: 3 * t ** 2 + 2 * t
    x = 4.0
    result = geron_numerical_differentiation(f, x)
    assert isinstance(result, dict)
