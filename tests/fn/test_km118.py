"""Tests for km118.kamath_ch8_rouge_n."""

import math

from morie.fn import _array_core as np

from morie.fn.km118 import kamath_ch8_rouge_n


def test_km118_basic():
    """Test basic functionality."""
    S = [["the", "cat", "sat", "on", "the", "mat"],
         ["a", "cat", "sat", "down"]]
    candidate = ["the", "cat", "sat", "on"]
    result = kamath_ch8_rouge_n(S, 1, candidate=candidate)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    assert "matched" in result
    assert "total_reference_ngrams" in result
    assert "gram_n" in result
    assert "n" in result
    assert result["gram_n"] == 1
    assert result["n"] == len(S)


def test_km118_edge():
    """Test edge cases."""
    S = [["the", "cat", "sat", "on", "the", "mat"]]
    candidate = ["the", "cat", "sat", "on", "the", "mat"]
    result = kamath_ch8_rouge_n(S, 2, candidate=candidate)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    assert "matched" in result
    assert "total_reference_ngrams" in result
    assert result["gram_n"] == 2
    assert result["n"] == 1
