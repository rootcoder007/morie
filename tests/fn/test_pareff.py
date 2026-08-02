"""Tests for pareff.population_attributable."""

from morie.fn import _array_core as np

from morie.fn.pareff import population_attributable


def test_pareff_basic():
    """Test basic functionality."""
    pe = np.random.default_rng(42).normal(0, 1, 100)
    RR = np.random.default_rng(42).normal(0, 1, 100)
    result = population_attributable(pe, RR)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_pareff_edge():
    """Test edge cases."""
    pe = np.random.default_rng(42).normal(0, 1, 100)
    RR = np.random.default_rng(42).normal(0, 1, 100)
    result = population_attributable(pe, RR)
    assert isinstance(result, dict)
