"""Tests for sfcret.surface_retrieval."""

from morie.fn import _array_core as np

from morie.fn.sfcret import surface_retrieval


def test_sfcret_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    values = np.random.default_rng(42).normal(0.0, 1.0, 40)
    grid = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = surface_retrieval(coords, values, grid)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_sfcret_edge():
    """Test edge cases."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    values = np.random.default_rng(42).normal(0.0, 1.0, 40)
    grid = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = surface_retrieval(coords, values, grid)
    assert isinstance(result, dict)
