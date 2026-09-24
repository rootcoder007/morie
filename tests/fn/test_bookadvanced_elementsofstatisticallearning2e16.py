"""Tests for bookadvanced_elementsofstatisticallearning2e16.bookadvanced_elementsofstatisticallearning_chapter_2_equation_16."""

import math

from morie.fn import _array_core as np

from morie.fn.bookadvanced_elementsofstatisticallearning2e16 import (
    bookadvanced_elementsofstatisticallearning_chapter_2_equation_16,
)


def test_bookadvanced_elementsofstatisticallearning2e16_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_16(x, y)
    assert isinstance(result, dict)
    assert len(result) > 0
    assert any(
        isinstance(v, (int, float)) and math.isfinite(v) for v in result.values()
    )


def test_bookadvanced_elementsofstatisticallearning2e16_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 10)
    y = rng.normal(0, 1, 10)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_16(x, y)
    assert isinstance(result, dict)
    assert len(result) > 0
    assert any(
        isinstance(v, (int, float)) and math.isfinite(v) for v in result.values()
    )
