"""Tests for statistical_methods_for_spatial_data_analysis1e28.statistical_methods_for_spatial_data_analysis_chapter_1_equation_28."""

from morie.fn import _array_core as np

from morie.fn.statistical_methods_for_spatial_data_analysis1e28 import (
    statistical_methods_for_spatial_data_analysis_chapter_1_equation_28,
)


def test_statistical_methods_for_spatial_data_analysis1e28_basic():
    """Test basic functionality."""
    n = 5
    rho = 0.1
    result = statistical_methods_for_spatial_data_analysis_chapter_1_equation_28(n, rho)
    assert isinstance(result, dict)
    assert "sigma" in result or "sigma" in result


def test_statistical_methods_for_spatial_data_analysis1e28_edge():
    """Test edge cases."""
    n = 5
    rho = 0.1
    result = statistical_methods_for_spatial_data_analysis_chapter_1_equation_28(n, rho)
    assert isinstance(result, dict)
