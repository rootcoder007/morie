"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r4e2.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_4_equation_2."""

from morie.fn import _array_core as np

from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r4e2 import (
    the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_4_equation_2,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r4e2_basic():
    """Test basic functionality."""
    z_h = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_4_equation_2(z_h)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r4e2_edge():
    """Test edge cases."""
    z_h = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_4_equation_2(z_h)
    assert isinstance(result, dict)
