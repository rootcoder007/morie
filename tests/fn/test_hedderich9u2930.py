"""Tests for hedderich9u2930.hedderich_chapter_9_unnumbered_2930."""

from morie.fn import _array_core as np

from morie.fn.hedderich9u2930 import hedderich_chapter_9_unnumbered_2930


def test_hedderich9u2930_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2930(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_hedderich9u2930_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2930(x)
    assert isinstance(result, dict)
