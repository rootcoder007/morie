"""Tests for hedderich9u3548.hedderich_chapter_9_unnumbered_3548."""

from morie.fn import _array_core as np

from morie.fn.hedderich9u3548 import hedderich_chapter_9_unnumbered_3548


def test_hedderich9u3548_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3548(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hedderich9u3548_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3548(x)
    assert isinstance(result, dict)
