"""Tests for bookadvanced_elementsofstatisticallearning2e28.bookadvanced_elementsofstatisticallearning_chapter_2_equation_28."""

import math
import pytest

from morie.fn import _array_core as np

from morie.fn.bookadvanced_elementsofstatisticallearning2e28 import (
    bookadvanced_elementsofstatisticallearning_chapter_2_equation_28,
)


def test_bookadvanced_elementsofstatisticallearning2e28_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    beta = rng.normal(0, 1, p)
    noise = rng.normal(0, 1, n)
    y = [sum(X[i][j] * beta[j] for j in range(p)) + noise[i] for i in range(n)]
    x0 = rng.normal(0, 1, p)
    with pytest.warns(DeprecationWarning):
        result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_28(X, y, x0)
    assert isinstance(result, dict)
    assert len(result) >= 1
    for v in result.values():
        if isinstance(v, (int, float)):
            assert math.isfinite(v)


def test_bookadvanced_elementsofstatisticallearning2e28_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p = 5, 2
    X = rng.normal(0, 1, (n, p))
    beta = rng.normal(0, 1, p)
    noise = rng.normal(0, 1, n)
    y = [sum(X[i][j] * beta[j] for j in range(p)) + noise[i] for i in range(n)]
    x0 = rng.normal(0, 1, p)
    with pytest.warns(DeprecationWarning):
        result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_28(X, y, x0)
    assert isinstance(result, dict)
    assert len(result) >= 1
    for v in result.values():
        if isinstance(v, (int, float)):
            assert math.isfinite(v)
