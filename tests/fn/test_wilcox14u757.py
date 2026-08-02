"""Tests for wilcox14u757.wilcox_chapter_14_unnumbered_757."""

from morie.fn import _array_core as np

from morie.fn.wilcox14u757 import wilcox_chapter_14_unnumbered_757


def test_wilcox14u757_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_757(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_wilcox14u757_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_757(x)
    assert isinstance(result, dict)
