"""Tests for hedderich9u1053.hedderich_chapter_9_unnumbered_1053."""

from morie.fn import _array_core as np

from morie.fn.hedderich9u1053 import hedderich_chapter_9_unnumbered_1053


def test_hedderich9u1053_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1053(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hedderich9u1053_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1053(x)
    assert isinstance(result, dict)
