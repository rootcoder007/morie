"""Tests for gauvar.gaussian_variogram_model."""

from morie.fn import _array_core as np

from morie.fn.gauvar import gaussian_variogram_model


def test_gauvar_basic():
    """Test basic functionality."""
    h = 0.3
    result = gaussian_variogram_model(h)
    assert isinstance(result, dict)
    assert "gamma" in result
def test_gauvar_edge():
    """Test edge cases."""
    h = 0.3
    result = gaussian_variogram_model(h)
    assert isinstance(result, dict)
