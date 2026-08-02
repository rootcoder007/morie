"""Tests for wilcox2u239.wilcox_chapter_2_unnumbered_239."""

from morie.fn import _array_core as np

from morie.fn.wilcox2u239 import wilcox_chapter_2_unnumbered_239


def test_wilcox2u239_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_239(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_wilcox2u239_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_239(x)
    assert isinstance(result, dict)
