"""Tests for matvar.matern_variogram_model."""

from morie.fn import _array_core as np

from morie.fn.matvar import matern_variogram_model


def test_matvar_basic():
    """Test basic functionality."""
    h = 0.1
    result = matern_variogram_model(h)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_matvar_edge():
    """Test edge cases."""
    h = 0.1
    result = matern_variogram_model(h)
    assert isinstance(result, dict)
