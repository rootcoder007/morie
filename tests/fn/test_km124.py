"""Tests for km124.kamath_ch8_ngram_embedding."""

from morie.fn import _array_core as np

from morie.fn.km124 import kamath_ch8_ngram_embedding


def test_km124_basic():
    """Test basic functionality as per docstring example."""
    x = [1.0, 2.0, 3.0]
    i = 1
    n = 2
    result = kamath_ch8_ngram_embedding(x, i, n)
    # Check returned keys
    assert "estimate" in result
    assert "embedding" in result
    assert "window" in result
    # Verify core values
    assert result["i"] == 1
    assert result["n"] == 2
    assert result["estimate"] == 5.0  # 2 + 3
    assert result["embedding"] == [5.0]
    assert result["window"] == [2.0, 3.0]


def test_km124_edge():
    """Test edge cases: n=1 (smallest valid n) and 2-D input."""
    # n = 1, 1-D input
    x = [5.0, 6.0, 7.0]
    result = kamath_ch8_ngram_embedding(x, 2, 1)
    assert "estimate" in result
    assert result["i"] == 2
    assert result["n"] == 1
    assert result["estimate"] == 7.0
    assert result["embedding"] == [7.0]
    assert result["window"] == [7.0]

    # 2-D input: each token has a 2-element vector
    x2 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    result2 = kamath_ch8_ngram_embedding(x2, 1, 2)
    assert "estimate" in result2
    assert result2["i"] == 1
    assert result2["n"] == 2
    # Window rows 1 and 2: [3,4] + [5,6] => [8, 10]
    assert result2["estimate"] == [8.0, 10.0]
    assert result2["embedding"] == [8.0, 10.0]
    # Window flattened row-major
    assert result2["window"] == [3.0, 4.0, 5.0, 6.0]
