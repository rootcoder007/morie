"""Tests for ht_mean.ht_mean."""

from morie.fn import _array_core as np

from morie.fn.ht_mean import (
    ht_mean,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r2e4_basic():
    """Test basic functionality."""
    z = 0.5
    pi = 0.5
    n_population = 0.5
    result = ht_mean(z, pi, n_population)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r2e4_edge():
    """Test edge cases."""
    z = 0.5
    pi = 0.5
    n_population = 0.5
    result = ht_mean(z, pi, n_population)
    assert isinstance(result, dict)
