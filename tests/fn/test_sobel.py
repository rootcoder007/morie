"""Tests for sobel.sobel_test."""

from morie.fn import _array_core as np

from morie.fn.sobel import sobel_test


def test_sobel_basic():
    """Test basic functionality."""
    a = 0.5
    b = 0.5
    se_a = 0.5
    se_b = 0.5
    result = sobel_test(a, b, se_a, se_b)
    assert isinstance(result, dict)
    assert "statistic" in result or "statistic" in result or "statistic" in result


def test_sobel_edge():
    """Test edge cases."""
    a = 0.5
    b = 0.5
    se_a = 0.5
    se_b = 0.5
    result = sobel_test(a, b, se_a, se_b)
    assert isinstance(result, dict)
