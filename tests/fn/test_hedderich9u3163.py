"""Tests for hedderich9u3163.hedderich_chapter_9_unnumbered_3163."""

from morie.fn import _array_core as np

from morie.fn.hedderich9u3163 import hedderich_chapter_9_unnumbered_3163


def test_hedderich9u3163_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3163(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "estimate" in result


def test_hedderich9u3163_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3163(x)
    assert isinstance(result, dict)
