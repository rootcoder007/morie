"""Tests for bookadvanced_elementsofstatisticallearning2e26.bookadvanced_elementsofstatisticallearning_chapter_2_equation_26."""

import pytest
from morie.fn import _array_core as np

from morie.fn.bookadvanced_elementsofstatisticallearning2e26 import (
    bookadvanced_elementsofstatisticallearning_chapter_2_equation_26,
)


def test_bookadvanced_elementsofstatisticallearning2e26_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 40)
    y = rng.normal(0, 1, 40)
    with pytest.warns(DeprecationWarning):
        result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_26(x, y)
    assert isinstance(result, dict)


def test_bookadvanced_elementsofstatisticallearning2e26_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 10)
    y = rng.normal(0, 1, 10)
    with pytest.warns(DeprecationWarning):
        result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_26(x, y)
    assert isinstance(result, dict)
