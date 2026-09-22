"""Tests for advcmp.advanced_composition."""

from morie.fn import _array_core as np

from morie.fn.advcmp import advanced_composition


def test_advcmp_basic():
    """Test basic functionality."""
    epsilon = 1e-6
    result = advanced_composition(epsilon)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_advcmp_edge():
    """Test edge cases."""
    epsilon = 1e-6
    result = advanced_composition(epsilon)
    assert isinstance(result, dict)
