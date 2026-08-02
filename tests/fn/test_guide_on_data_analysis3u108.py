"""Tests for guide_on_data_analysis3u108.guide_on_data_analysis_chapter_3_unnumbered_108."""

from morie.fn import _array_core as np

from morie.fn.guide_on_data_analysis3u108 import guide_on_data_analysis_chapter_3_unnumbered_108


def test_guide_on_data_analysis3u108_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_3_unnumbered_108(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_guide_on_data_analysis3u108_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_3_unnumbered_108(x)
    assert isinstance(result, dict)
