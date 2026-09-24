"""Tests for statistical_methods_for_spatial_data_analysis1e2.statistical_methods_for_spatial_data_analysis_chapter_1_equation_2."""

from morie.fn import _array_core as np

from morie.fn.statistical_methods_for_spatial_data_analysis1e2 import (
    statistical_methods_for_spatial_data_analysis_chapter_1_equation_2,
)


def test_statistical_methods_for_spatial_data_analysis1e2_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_equation_2(coords, x, y)
    assert isinstance(result, dict)
    assert "beta" in result or "beta" in result


def test_statistical_methods_for_spatial_data_analysis1e2_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_equation_2(coords, x, y)
    assert isinstance(result, dict)
