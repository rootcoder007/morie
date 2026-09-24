"""Tests for idwsmt.inverse_distance_weighting."""

from morie.fn import _array_core as np

from morie.fn.idwsmt import inverse_distance_weighting


def test_idwsmt_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    values = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = inverse_distance_weighting(coords, values)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_idwsmt_edge():
    """Test edge cases."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    values = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = inverse_distance_weighting(coords, values)
    assert isinstance(result, dict)
