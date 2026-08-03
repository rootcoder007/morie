"""Tests for ols_beta.ols_beta."""

from morie.fn import _array_core as np

from morie.fn.ols_beta import (
    ols_beta,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r20e2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ols_beta(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r20e2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ols_beta(x)
    assert isinstance(result, dict)
