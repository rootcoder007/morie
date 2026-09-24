"""Tests for km125.kamath_ch8_ngram_weight."""

from morie.fn import _array_core as np

from morie.fn.km125 import kamath_ch8_ngram_weight


def test_km125_basic():
    """Test basic functionality."""
    x = [[1.0, 2.0], [3.0, 4.0]]
    result = kamath_ch8_ngram_weight(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_km125_edge():
    """Test edge cases."""
    x = [[1.0, 2.0], [3.0, 4.0]]
    result = kamath_ch8_ngram_weight(x)
    assert isinstance(result, dict)
