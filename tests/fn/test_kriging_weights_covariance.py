"""Tests for kriging_weights_covariance.kriging_weights_covariance."""

from morie.fn import _array_core as np

from morie.fn.kriging_weights_covariance import (
    kriging_weights_covariance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kriging_weights_covariance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kriging_weights_covariance(x)
    assert isinstance(result, dict)
