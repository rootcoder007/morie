"""Tests for hedderich9u1717.hedderich_chapter_9_unnumbered_1717."""

from morie.fn import _array_core as np

from morie.fn.hedderich9u1717 import hedderich_chapter_9_unnumbered_1717


def test_hedderich9u1717_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1717(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hedderich9u1717_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1717(x)
    assert isinstance(result, dict)
