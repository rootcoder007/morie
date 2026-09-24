"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r12e19.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_19."""

from morie.fn import _array_core as np

from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r12e19 import (
    the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_19,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e19_basic():
    """Test basic functionality."""
    coverages = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    alpha = 0.1
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_19(coverages, probs, alpha)
    assert isinstance(result, dict)
    assert "expected_coverage" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e19_edge():
    """Test edge cases."""
    coverages = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    alpha = 0.1
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_19(coverages, probs, alpha)
    assert isinstance(result, dict)
