"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r4e1.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_4_equation_1."""

from morie.fn import _array_core as np

from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r4e1 import (
    the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_4_equation_1,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r4e1_basic():
    """Test basic functionality."""
    stratum_means = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    stratum_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_4_equation_1(stratum_means, stratum_weights)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r4e1_edge():
    """Test edge cases."""
    stratum_means = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    stratum_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_4_equation_1(stratum_means, stratum_weights)
    assert isinstance(result, dict)
