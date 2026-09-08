"""Tests for estimation_adjusted_criterion.estimation_adjusted_criterion."""

from morie.fn import _array_core as np

from morie.fn.estimation_adjusted_criterion import (
    estimation_adjusted_criterion,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e6_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = estimation_adjusted_criterion(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e6_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = estimation_adjusted_criterion(x)
    assert isinstance(result, dict)
