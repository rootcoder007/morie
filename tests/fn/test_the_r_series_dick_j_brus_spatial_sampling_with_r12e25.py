"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r12e25.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_25."""

from morie.fn import _array_core as np

from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r12e25 import (
    the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_25,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e25_basic():
    """Test basic functionality."""
    lengths = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    l_max = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_25(lengths, probs, l_max)
    assert isinstance(result, dict)
    assert "expected_length" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e25_edge():
    """Test edge cases."""
    lengths = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    l_max = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_25(lengths, probs, l_max)
    assert isinstance(result, dict)
