"""Tests for mean_semivariance_stsi_variance.mean_semivariance_stsi_variance."""

from morie.fn import _array_core as np

from morie.fn.mean_semivariance_stsi_variance import (
    mean_semivariance_stsi_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e5_basic():
    """Test basic functionality."""
    gamma_bar_h = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    n_h = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = mean_semivariance_stsi_variance(gamma_bar_h, weights, n_h)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e5_edge():
    """Test edge cases."""
    gamma_bar_h = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    n_h = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = mean_semivariance_stsi_variance(gamma_bar_h, weights, n_h)
    assert isinstance(result, dict)
