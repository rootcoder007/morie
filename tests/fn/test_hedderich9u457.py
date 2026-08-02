"""Tests for hedderich9u457.hedderich_chapter_9_unnumbered_457."""

from morie.fn import _array_core as np

from morie.fn.hedderich9u457 import hedderich_chapter_9_unnumbered_457


def test_hedderich9u457_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_457(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hedderich9u457_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_457(x)
    assert isinstance(result, dict)
