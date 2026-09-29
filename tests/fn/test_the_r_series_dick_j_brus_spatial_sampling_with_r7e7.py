"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r7e7.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_7."""

from morie.fn import _array_core as np
from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r7e7 import (
    the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_7,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r7e7_basic():
    """Test basic functionality."""
    primary_unit_means = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_7(primary_unit_means)
    assert isinstance(result, dict)
    assert "s2_psu" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r7e7_edge():
    """Test edge cases."""
    primary_unit_means = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_7(primary_unit_means)
    assert isinstance(result, dict)


def test_two_stage_variance_estimator():
    import pytest

    m = [2.0, 4.5, 3.0, 7.5]
    mb = sum(m) / 4
    s2 = sum((v - mb) ** 2 for v in m) / 3
    r = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_7(m)
    assert r["value"] == pytest.approx(s2 / 4, rel=1e-14)
    assert r["s2_psu"] == pytest.approx(s2, rel=1e-14)
