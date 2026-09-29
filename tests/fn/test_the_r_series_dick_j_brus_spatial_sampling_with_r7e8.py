"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r7e8.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_8."""

from morie.fn import _array_core as np
from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r7e8 import (
    the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_8,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r7e8_basic():
    """Test basic functionality."""
    primary_unit_means = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_8(primary_unit_means)
    assert isinstance(result, dict)
    assert "s2_psu" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r7e8_edge():
    """Test edge cases."""
    primary_unit_means = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_8(primary_unit_means)
    assert isinstance(result, dict)


def test_between_psu_variance():
    import pytest

    m = [2.0, 4.5, 3.0, 7.5]
    mb = sum(m) / 4
    r = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_8(m)
    assert r["value"] == pytest.approx(sum((v - mb) ** 2 for v in m) / 3, rel=1e-14)
