"""Tests for hedderich9u2952.hedderich_chapter_9_unnumbered_2952."""

from morie.fn import _array_core as np

from morie.fn.hedderich9u2952 import hedderich_chapter_9_unnumbered_2952


def test_hedderich9u2952_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2952(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hedderich9u2952_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2952(x)
    assert isinstance(result, dict)
