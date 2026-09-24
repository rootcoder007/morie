"""Tests for nndist.nearest_neighbor_distance."""

from morie.fn import _array_core as np

from morie.fn.nndist import nearest_neighbor_distance


def test_nndist_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    r_grid = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = nearest_neighbor_distance(coords, r_grid)
    assert isinstance(result, dict)
    assert "estimate" in result or "G" in result


def test_nndist_edge():
    """Test edge cases."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    r_grid = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = nearest_neighbor_distance(coords, r_grid)
    assert isinstance(result, dict)
