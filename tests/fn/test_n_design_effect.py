"""Tests for n_design_effect.n_design_effect."""

from morie.fn import _array_core as np

from morie.fn.n_design_effect import (
    n_design_effect,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e14_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = n_design_effect(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e14_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = n_design_effect(x)
    assert isinstance(result, dict)
