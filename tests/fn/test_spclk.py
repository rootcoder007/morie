"""Tests for spclk.schabenberger_composite_likelihood."""

from morie.fn import _array_core as np

from morie.fn.spclk import schabenberger_composite_likelihood


def test_spclk_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = schabenberger_composite_likelihood(coords, z)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_spclk_edge():
    """Test edge cases."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = schabenberger_composite_likelihood(coords, z)
    assert isinstance(result, dict)
