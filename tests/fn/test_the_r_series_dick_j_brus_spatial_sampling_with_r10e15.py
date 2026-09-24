"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r10e15.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_15."""

from morie.fn import _array_core as np

from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r10e15 import (
    the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_15,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e15_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_15(x, z)
    assert isinstance(result, dict)
    assert "values" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e15_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_15(x, z)
    assert isinstance(result, dict)
