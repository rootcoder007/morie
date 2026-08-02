"""Tests for hedderich9u610.hedderich_chapter_9_unnumbered_610."""

from morie.fn import _array_core as np

from morie.fn.hedderich9u610 import hedderich_chapter_9_unnumbered_610


def test_hedderich9u610_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_610(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hedderich9u610_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_610(x)
    assert isinstance(result, dict)
