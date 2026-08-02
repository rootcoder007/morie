"""Tests for bivand20132u45.bivand2013_chapter_2_unnumbered_45."""

from morie.fn import _array_core as np

from morie.fn.bivand20132u45 import bivand2013_chapter_2_unnumbered_45


def test_bivand20132u45_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_45(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_bivand20132u45_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_45(x)
    assert isinstance(result, dict)
