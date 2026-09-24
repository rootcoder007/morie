"""Tests for regression_estimator_general.regression_estimator_general."""

from morie.fn import _array_core as np

from morie.fn.regression_estimator_general import (
    regression_estimator_general,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e8_basic():
    """Test basic functionality."""
    x_all = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    b_hat = 5
    z_sample = 5
    x_sample = 5
    pi_sample = 5
    n_population = 5
    result = regression_estimator_general(x_all, b_hat, z_sample, x_sample, pi_sample, n_population)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e8_edge():
    """Test edge cases."""
    x_all = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    b_hat = 5
    z_sample = 5
    x_sample = 5
    pi_sample = 5
    n_population = 5
    result = regression_estimator_general(x_all, b_hat, z_sample, x_sample, pi_sample, n_population)
    assert isinstance(result, dict)
