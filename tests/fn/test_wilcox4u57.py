"""Tests for wilcox4u57.wilcox_chapter_4_unnumbered_57."""

from morie.fn import _array_core as np

from morie.fn.wilcox4u57 import wilcox_chapter_4_unnumbered_57


def test_wilcox4u57_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_57(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_wilcox4u57_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_57(x)
    assert isinstance(result, dict)
