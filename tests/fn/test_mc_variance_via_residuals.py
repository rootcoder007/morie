"""Tests for mc_variance_via_residuals.mc_variance_via_residuals."""

from morie.fn import _array_core as np

from morie.fn.mc_variance_via_residuals import (
    mc_variance_via_residuals,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e42_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = mc_variance_via_residuals(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e42_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = mc_variance_via_residuals(x)
    assert isinstance(result, dict)
