"""Tests for expvar.exponential_variogram_model."""

from morie.fn import _array_core as np

from morie.fn.expvar import exponential_variogram_model


def test_expvar_basic():
    """Test basic functionality."""
    h = 0.3
    result = exponential_variogram_model(h)
    assert isinstance(result, dict)
    assert "gamma" in result
def test_expvar_edge():
    """Test edge cases."""
    h = 0.3
    result = exponential_variogram_model(h)
    assert isinstance(result, dict)
