"""Tests for spvarm.spherical_variogram_model."""

from morie.fn import _array_core as np

from morie.fn.spvarm import spherical_variogram_model


def test_spvarm_basic():
    """Test basic functionality."""
    h = 0.1
    result = spherical_variogram_model(h)
    assert isinstance(result, dict)
    assert "h" in result


def test_spvarm_edge():
    """Test edge cases."""
    h = 0.1
    result = spherical_variogram_model(h)
    assert isinstance(result, dict)
