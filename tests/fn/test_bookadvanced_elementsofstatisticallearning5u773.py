"""Tests for bookadvanced_elementsofstatisticallearning5u773.bookadvanced_elementsofstatisticallearning_chapter_5_unnumbered_773."""

from morie.fn import _array_core as np

from morie.fn.bookadvanced_elementsofstatisticallearning5u773 import (
    bookadvanced_elementsofstatisticallearning_chapter_5_unnumbered_773,
)


def test_bookadvanced_elementsofstatisticallearning5u773_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_5_unnumbered_773(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_bookadvanced_elementsofstatisticallearning5u773_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_5_unnumbered_773(x)
    assert isinstance(result, dict)
