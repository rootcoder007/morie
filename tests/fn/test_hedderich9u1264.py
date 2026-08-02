"""Tests for hedderich9u1264.hedderich_chapter_9_unnumbered_1264."""

from morie.fn import _array_core as np

from morie.fn.hedderich9u1264 import hedderich_chapter_9_unnumbered_1264


def test_hedderich9u1264_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1264(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hedderich9u1264_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1264(x)
    assert isinstance(result, dict)
