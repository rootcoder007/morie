"""Tests for mixed_calibration_mean.mixed_calibration_mean."""

from morie.fn import _array_core as np

from morie.fn.mixed_calibration_mean import (
    mixed_calibration_mean,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e36_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = mixed_calibration_mean(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e36_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = mixed_calibration_mean(x)
    assert isinstance(result, dict)
