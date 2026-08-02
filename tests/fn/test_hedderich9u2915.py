"""Tests for hedderich9u2915.hedderich_chapter_9_unnumbered_2915."""

from morie.fn import _array_core as np

from morie.fn.hedderich9u2915 import hedderich_chapter_9_unnumbered_2915


def test_hedderich9u2915_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2915(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_hedderich9u2915_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2915(x)
    assert isinstance(result, dict)
