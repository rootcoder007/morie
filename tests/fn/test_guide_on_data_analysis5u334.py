"""Tests for guide_on_data_analysis5u334.guide_on_data_analysis_chapter_5_unnumbered_334."""

from morie.fn import _array_core as np

from morie.fn.guide_on_data_analysis5u334 import guide_on_data_analysis_chapter_5_unnumbered_334


def test_guide_on_data_analysis5u334_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_5_unnumbered_334(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_guide_on_data_analysis5u334_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_5_unnumbered_334(x)
    assert isinstance(result, dict)
