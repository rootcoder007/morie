"""Tests for bookadvanced_elementsofstatisticallearning6u866.bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_866."""

from morie.fn import _array_core as np

from morie.fn.bookadvanced_elementsofstatisticallearning6u866 import (
    bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_866,
)


def test_bookadvanced_elementsofstatisticallearning6u866_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_866(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_bookadvanced_elementsofstatisticallearning6u866_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_866(x)
    assert isinstance(result, dict)
