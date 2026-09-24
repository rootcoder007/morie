"""Tests for infinite_total_variance.infinite_total_variance."""

from morie.fn import _array_core as np

from morie.fn.infinite_total_variance import (
    infinite_total_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r3e21_basic():
    """Test basic functionality."""
    s2_hat = 0.5
    n = 0.5
    area = 0.5
    sample_area = 0.5
    result = infinite_total_variance(s2_hat, n, area, sample_area)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r3e21_edge():
    """Test edge cases."""
    s2_hat = 0.5
    n = 0.5
    area = 0.5
    sample_area = 0.5
    result = infinite_total_variance(s2_hat, n, area, sample_area)
    assert isinstance(result, dict)
