"""Tests for mixed_calibration_intercept.mixed_calibration_intercept."""

from morie.fn import _array_core as np

from morie.fn.mixed_calibration_intercept import (
    mixed_calibration_intercept,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e38_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = mixed_calibration_intercept(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e38_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = mixed_calibration_intercept(x)
    assert isinstance(result, dict)
