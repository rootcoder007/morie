"""Tests for bookadvanced_elementsofstatisticallearning2e45.bookadvanced_elementsofstatisticallearning_chapter_2_equation_45."""

import math

from morie.fn import _array_core as np

from morie.fn.bookadvanced_elementsofstatisticallearning2e45 import (
    bookadvanced_elementsofstatisticallearning_chapter_2_equation_45,
)


def test_bookadvanced_elementsofstatisticallearning2e45_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p, M = 40, 3, 5
    X = rng.normal(0, 1, (n, p))
    alpha = rng.normal(0, 1, (p, M))
    b = rng.normal(0, 1, M)
    beta = rng.normal(0, 1, M)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_45(
        X, alpha, b, beta=beta
    )
    assert isinstance(result, dict)
    for key in ("estimate", "fitted", "hidden", "beta", "rss", "n", "p", "M", "method"):
        assert key in result
    assert result["n"] == n
    assert result["p"] == p
    assert result["M"] == M
    assert math.isfinite(result["estimate"])
    assert len(result["fitted"]) == n
    assert len(result["beta"]) == M
    assert len(result["hidden"]) == n
    assert len(result["hidden"][0]) == M


def test_bookadvanced_elementsofstatisticallearning2e45_edge():
    """Test edge cases."""
    rng = np.random.default_rng(7)
    n, p, M = 40, 3, 4
    X = rng.normal(0, 1, (n, p))
    alpha = rng.normal(0, 1, (p, M))
    b = rng.normal(0, 1, M)
    y = rng.normal(0, 1, n)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_45(
        X, alpha, b, y=y
    )
    assert isinstance(result, dict)
    assert result["n"] == n
    assert result["p"] == p
    assert result["M"] == M
    assert math.isfinite(result["rss"])
    assert math.isfinite(result["estimate"])
    assert len(result["fitted"]) == n
    assert len(result["beta"]) == M
    assert len(result["hidden"]) == n
    assert len(result["hidden"][0]) == M
    rss = sum((y[i] - result["fitted"][i]) ** 2 for i in range(n))
    assert math.isclose(result["rss"], rss)
