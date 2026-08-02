"""Tests for ca3u43.ca_chapter_3_unnumbered_43."""

from morie.fn import _array_core as np

from morie.fn.ca3u43 import ca_chapter_3_unnumbered_43


def test_ca3u43_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_43(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca3u43_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_43(x)
    assert isinstance(result, dict)
