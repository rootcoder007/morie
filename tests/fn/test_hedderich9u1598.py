"""Tests for hedderich9u1598.hedderich_chapter_9_unnumbered_1598."""

from morie.fn import _array_core as np

from morie.fn.hedderich9u1598 import hedderich_chapter_9_unnumbered_1598


def test_hedderich9u1598_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1598(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hedderich9u1598_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1598(x)
    assert isinstance(result, dict)
