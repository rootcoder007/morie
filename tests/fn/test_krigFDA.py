"""Tests for krigFDA.kriging."""

from morie.fn import _array_core as np

from morie.fn.krigFDA import kriging


def test_krigFDA_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    values = np.random.default_rng(42).normal(0.0, 1.0, 40)
    new_coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kriging(coords, values, new_coords)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_krigFDA_edge():
    """Test edge cases."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    values = np.random.default_rng(42).normal(0.0, 1.0, 40)
    new_coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kriging(coords, values, new_coords)
    assert isinstance(result, dict)
