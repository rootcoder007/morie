"""Tests for pareff.population_attributable."""

from morie.fn import _array_core as np

from morie.fn.pareff import population_attributable


def test_pareff_basic():
    """Test basic functionality."""
    pe = 0.1
    RR = 0.1
    result = population_attributable(pe, RR)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_pareff_edge():
    """Test edge cases."""
    pe = 0.1
    RR = 0.1
    result = population_attributable(pe, RR)
    assert isinstance(result, dict)
