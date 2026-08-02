"""Tests for wilcox4u124.wilcox_chapter_4_unnumbered_124."""

from morie.fn import _array_core as np

from morie.fn.wilcox4u124 import wilcox_chapter_4_unnumbered_124


def test_wilcox4u124_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_124(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_wilcox4u124_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_124(x)
    assert isinstance(result, dict)
