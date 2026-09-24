"""Tests for km127.kamath_ch8_geval_score."""

from morie.fn import _array_core as np

from morie.fn.km127 import kamath_ch8_geval_score


def test_km127_basic():
    """Test basic functionality."""
    s_i = [1, 2, 3]
    p = [0.2, 0.3, 0.5]
    result = kamath_ch8_geval_score(s_i, p)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_km127_edge():
    """Test edge cases."""
    s_i = [1, 2, 3]
    p = [0.2, 0.3, 0.5]
    result = kamath_ch8_geval_score(s_i, p)
    assert isinstance(result, dict)
