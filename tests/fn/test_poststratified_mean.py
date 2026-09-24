"""Tests for poststratified_mean.poststratified_mean."""

from morie.fn import _array_core as np

from morie.fn.poststratified_mean import (
    poststratified_mean,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e32_basic():
    """Test basic functionality."""
    group_means_sample = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    group_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = poststratified_mean(group_means_sample, group_weights)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e32_edge():
    """Test edge cases."""
    group_means_sample = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    group_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = poststratified_mean(group_means_sample, group_weights)
    assert isinstance(result, dict)
