"""Tests for polyak.polyak_target."""

from morie.fn import _array_core as np

from morie.fn.polyak import polyak_target


def test_polyak_basic():
    """Test basic functionality."""
    iterates = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = polyak_target(iterates)
    assert isinstance(result, dict)
    assert "average" in result


def test_polyak_edge():
    """Test edge cases."""
    iterates = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = polyak_target(iterates)
    assert isinstance(result, dict)
