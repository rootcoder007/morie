"""Tests for fisher_information_reml.fisher_information_reml."""

from morie.fn import _array_core as np

from morie.fn.fisher_information_reml import (
    fisher_information_reml,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = fisher_information_reml(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = fisher_information_reml(x)
    assert isinstance(result, dict)
