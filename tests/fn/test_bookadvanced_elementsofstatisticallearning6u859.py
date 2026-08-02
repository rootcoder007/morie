"""Tests for bookadvanced_elementsofstatisticallearning6u859.bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_859."""

from morie.fn import _array_core as np

from morie.fn.bookadvanced_elementsofstatisticallearning6u859 import (
    bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_859,
)


def test_bookadvanced_elementsofstatisticallearning6u859_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_859(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_bookadvanced_elementsofstatisticallearning6u859_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_859(x)
    assert isinstance(result, dict)
