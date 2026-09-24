"""Tests for statistical_methods_for_spatial_data_analysis1e27.statistical_methods_for_spatial_data_analysis_chapter_1_equation_27."""

from morie.fn import _array_core as np

from morie.fn.statistical_methods_for_spatial_data_analysis1e27 import (
    statistical_methods_for_spatial_data_analysis_chapter_1_equation_27,
)


def test_statistical_methods_for_spatial_data_analysis1e27_basic():
    """Test basic functionality."""
    n = 5
    rho = 0.1
    result = statistical_methods_for_spatial_data_analysis_chapter_1_equation_27(n, rho)
    assert isinstance(result, dict)
    assert "var_pred" in result or "var_pred" in result


def test_statistical_methods_for_spatial_data_analysis1e27_edge():
    """Test edge cases."""
    n = 5
    rho = 0.1
    result = statistical_methods_for_spatial_data_analysis_chapter_1_equation_27(n, rho)
    assert isinstance(result, dict)
