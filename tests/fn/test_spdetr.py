"""Tests for spdetr.spatial_detrending."""

from morie.fn import _array_core as np

from morie.fn.spdetr import spatial_detrending


def test_spdetr_basic():
    """Test basic functionality."""
    values = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = spatial_detrending(values)
    assert isinstance(result, dict)
    assert "overall" in result


def test_spdetr_edge():
    """Test edge cases."""
    values = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = spatial_detrending(values)
    assert isinstance(result, dict)
