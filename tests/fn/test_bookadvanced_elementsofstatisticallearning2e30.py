"""Tests for bookadvanced_elementsofstatisticallearning2e30.bookadvanced_elementsofstatisticallearning_chapter_2_equation_30."""

from morie.fn import _array_core as np

from morie.fn.bookadvanced_elementsofstatisticallearning2e30 import (
    bookadvanced_elementsofstatisticallearning_chapter_2_equation_30,
)


def test_bookadvanced_elementsofstatisticallearning2e30_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 40)
    y = rng.normal(0, 1, 40)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_30(x, y)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_bookadvanced_elementsofstatisticallearning2e30_edge():
    """Test edge cases."""
    rng = np.random.default_rng(123)
    x = rng.normal(0, 1, 10)
    y = rng.normal(0, 1, 10)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_30(x, y)
    assert isinstance(result, dict)
    assert len(result) > 0
