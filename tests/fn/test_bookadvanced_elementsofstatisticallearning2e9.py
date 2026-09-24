"""Tests for bookadvanced_elementsofstatisticallearning2e9.bookadvanced_elementsofstatisticallearning_chapter_2_equation_9."""

from morie.fn import _array_core as np

from morie.fn.bookadvanced_elementsofstatisticallearning2e9 import (
    bookadvanced_elementsofstatisticallearning_chapter_2_equation_9,
)


def test_bookadvanced_elementsofstatisticallearning2e9_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_9(x, y)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_bookadvanced_elementsofstatisticallearning2e9_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    x = rng.normal(0, 1, 50)
    y = rng.normal(0, 1, 50)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_9(x, y)
    assert isinstance(result, dict)
