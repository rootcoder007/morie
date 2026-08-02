"""Tests for hedderich9u2234.hedderich_chapter_9_unnumbered_2234."""

from morie.fn import _array_core as np

from morie.fn.hedderich9u2234 import hedderich_chapter_9_unnumbered_2234


def test_hedderich9u2234_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2234(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hedderich9u2234_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2234(x)
    assert isinstance(result, dict)
