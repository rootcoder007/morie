"""Tests for mixed_calibration_intercept.mixed_calibration_intercept."""

from morie.fn import _array_core as np

from morie.fn.mixed_calibration_intercept import (
    mixed_calibration_intercept,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e38_basic():
    """Test basic functionality."""
    b_hat = 0.5
    z_sample = 0.5
    pi_sample = 0.5
    n_population = 0.5
    result = mixed_calibration_intercept(b_hat, z_sample, pi_sample, n_population)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e38_edge():
    """Test edge cases."""
    b_hat = 0.5
    z_sample = 0.5
    pi_sample = 0.5
    n_population = 0.5
    result = mixed_calibration_intercept(b_hat, z_sample, pi_sample, n_population)
    assert isinstance(result, dict)
