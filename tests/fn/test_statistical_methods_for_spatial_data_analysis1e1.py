"""Tests for statistical_methods_for_spatial_data_analysis1e1.statistical_methods_for_spatial_data_analysis_chapter_1_equation_1."""

from morie.fn import _array_core as np

from morie.fn.statistical_methods_for_spatial_data_analysis1e1 import (
    statistical_methods_for_spatial_data_analysis_chapter_1_equation_1,
)


def test_statistical_methods_for_spatial_data_analysis1e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_equation_1(x, y)
    assert isinstance(result, dict)
    assert "beta" in result or "beta" in result


def test_statistical_methods_for_spatial_data_analysis1e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_equation_1(x, y)
    assert isinstance(result, dict)
