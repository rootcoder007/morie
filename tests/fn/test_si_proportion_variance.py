"""Tests for si_proportion_variance.si_proportion_variance."""

from morie.fn import _array_core as np

from morie.fn.si_proportion_variance import (
    si_proportion_variance,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r3e14_basic():
    """Test basic functionality."""
    p_hat = 0.5
    n = 5
    n_population = 5
    result = si_proportion_variance(p_hat, n, n_population)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r3e14_edge():
    """Test edge cases."""
    p_hat = 0.5
    n = 5
    n_population = 5
    result = si_proportion_variance(p_hat, n, n_population)
    assert isinstance(result, dict)
