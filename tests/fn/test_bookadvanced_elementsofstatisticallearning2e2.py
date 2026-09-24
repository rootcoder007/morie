"""Tests for bookadvanced_elementsofstatisticallearning2e2.bookadvanced_elementsofstatisticallearning_chapter_2_equation_2."""

from morie.fn import _array_core as np

from morie.fn.bookadvanced_elementsofstatisticallearning2e2 import (
    bookadvanced_elementsofstatisticallearning_chapter_2_equation_2,
)


def test_bookadvanced_elementsofstatisticallearning2e2_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 40)
    y = rng.normal(0, 1, 40)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_2(x, y)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_bookadvanced_elementsofstatisticallearning2e2_edge():
    """Test edge cases."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5)
    y = rng.normal(0, 1, 5)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_2(x, y)
    assert isinstance(result, dict)
    assert len(result) > 0
