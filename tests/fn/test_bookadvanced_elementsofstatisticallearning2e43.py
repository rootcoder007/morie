"""Tests for bookadvanced_elementsofstatisticallearning2e43.bookadvanced_elementsofstatisticallearning_chapter_2_equation_43."""

import math

from morie.fn import _array_core as np

from morie.fn.bookadvanced_elementsofstatisticallearning2e43 import (
    bookadvanced_elementsofstatisticallearning_chapter_2_equation_43,
)


def test_bookadvanced_elementsofstatisticallearning2e43_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 40)
    y = rng.normal(0, 1, 40)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_43(x, y)
    # basisexp returns a RichResult dict-like object summarising the fit
    assert isinstance(result, dict)
    assert result["n"] == 40
    assert "K" in result
    assert "rss" in result
    assert math.isfinite(result["rss"])
    assert result["rss"] >= 0


def test_bookadvanced_elementsofstatisticallearning2e43_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 5)
    y = rng.normal(0, 1, 5)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_43(x, y)
    assert isinstance(result, dict)
    assert result["n"] == 5
    assert "K" in result
    assert "rss" in result
    assert math.isfinite(result["rss"])
    assert result["rss"] >= 0
