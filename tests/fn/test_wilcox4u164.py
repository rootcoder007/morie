"""Tests for wilcox4u164.wilcox_chapter_4_unnumbered_164."""

from morie.fn import _array_core as np

from morie.fn.wilcox4u164 import wilcox_chapter_4_unnumbered_164


def test_wilcox4u164_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_164(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_wilcox4u164_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_164(x)
    assert isinstance(result, dict)
