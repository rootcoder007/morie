"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r10e3.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_3."""

from morie.fn import _array_core as np

from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r10e3 import (
    the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_3,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e3_basic():
    """Test basic functionality."""
    x_k = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    eps = 0.1
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_3(x_k, beta, eps)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e3_edge():
    """Test edge cases."""
    x_k = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    eps = 0.1
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_3(x_k, beta, eps)
    assert isinstance(result, dict)
