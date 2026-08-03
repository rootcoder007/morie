"""Tests for spatial_lag_reduced_form.spatial_lag_reduced_form."""

from morie.fn import _array_core as np

from morie.fn.spatial_lag_reduced_form import spatial_lag_reduced_form


def test_ca12e4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_lag_reduced_form(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca12e4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_lag_reduced_form(x)
    assert isinstance(result, dict)
