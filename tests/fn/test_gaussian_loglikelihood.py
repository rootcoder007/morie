"""Tests for gaussian_loglikelihood.gaussian_loglikelihood."""

from morie.fn import _array_core as np

from morie.fn.gaussian_loglikelihood import (
    gaussian_loglikelihood,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e23_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = gaussian_loglikelihood(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e23_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = gaussian_loglikelihood(x)
    assert isinstance(result, dict)
