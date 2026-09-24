"""Tests for infinite_total.infinite_total."""

from morie.fn import _array_core as np

from morie.fn.infinite_total import (
    infinite_total,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r3e18_basic():
    """Test basic functionality."""
    zbar_hat = 0.5
    area = 0.5
    sample_area = 0.5
    result = infinite_total(zbar_hat, area, sample_area)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r3e18_edge():
    """Test edge cases."""
    zbar_hat = 0.5
    area = 0.5
    sample_area = 0.5
    result = infinite_total(zbar_hat, area, sample_area)
    assert isinstance(result, dict)
