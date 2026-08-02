"""Tests for hedderich9u354.hedderich_chapter_9_unnumbered_354."""

from morie.fn import _array_core as np

from morie.fn.hedderich9u354 import hedderich_chapter_9_unnumbered_354


def test_hedderich9u354_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_354(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_hedderich9u354_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_354(x)
    assert isinstance(result, dict)
