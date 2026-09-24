"""Tests for noncentrality_delta_generic.noncentrality_delta_generic."""

from morie.fn import _array_core as np

from morie.fn.noncentrality_delta_generic import noncentrality_delta_generic


def test_ca8e1_basic():
    """Test basic functionality."""
    mean_population = 0.5
    mean_null = 0.5
    result = noncentrality_delta_generic(mean_population, mean_null)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca8e1_edge():
    """Test edge cases."""
    mean_population = 0.5
    mean_null = 0.5
    result = noncentrality_delta_generic(mean_population, mean_null)
    assert isinstance(result, dict)
