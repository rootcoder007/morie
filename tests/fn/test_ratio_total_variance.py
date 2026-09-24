"""Tests for ratio_total_variance.ratio_total_variance."""

from morie.fn import _array_core as np

from morie.fn.ratio_total_variance import (
    ratio_total_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e25_basic():
    """Test basic functionality."""
    e = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    n = 5
    n_population = 5
    result = ratio_total_variance(e, n, n_population)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e25_edge():
    """Test edge cases."""
    e = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    n = 5
    n_population = 5
    result = ratio_total_variance(e, n, n_population)
    assert isinstance(result, dict)
