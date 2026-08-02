"""Tests for guide_on_data_analysis5u312.guide_on_data_analysis_chapter_5_unnumbered_312."""

from morie.fn import _array_core as np

from morie.fn.guide_on_data_analysis5u312 import guide_on_data_analysis_chapter_5_unnumbered_312


def test_guide_on_data_analysis5u312_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_5_unnumbered_312(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_guide_on_data_analysis5u312_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_5_unnumbered_312(x)
    assert isinstance(result, dict)
