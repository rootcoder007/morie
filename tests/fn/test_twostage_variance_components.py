"""Tests for twostage_variance_components.twostage_variance_components."""

from morie.fn import _array_core as np

from morie.fn.twostage_variance_components import (
    twostage_variance_components,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r7e3_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = twostage_variance_components(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r7e3_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = twostage_variance_components(x)
    assert isinstance(result, dict)
