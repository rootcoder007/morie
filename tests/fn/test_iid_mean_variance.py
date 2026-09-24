"""Tests for iid_mean_variance.iid_mean_variance."""

from morie.fn import _array_core as np

from morie.fn.iid_mean_variance import (
    iid_mean_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r26e2_basic():
    """Test basic functionality."""
    sigma2 = 0.5
    n = 0.5
    result = iid_mean_variance(sigma2, n)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r26e2_edge():
    """Test edge cases."""
    sigma2 = 0.5
    n = 0.5
    result = iid_mean_variance(sigma2, n)
    assert isinstance(result, dict)
