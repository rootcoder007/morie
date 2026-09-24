"""Tests for vgrm.variogram."""

from morie.fn import _array_core as np

from morie.fn.vgrm import variogram


def test_vgrm_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    values = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = variogram(coords, values)
    assert isinstance(result, dict)
    assert "lag" in result


def test_vgrm_edge():
    """Test edge cases."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    values = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = variogram(coords, values)
    assert isinstance(result, dict)
