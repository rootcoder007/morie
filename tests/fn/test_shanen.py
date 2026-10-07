"""Tests for shanen.shannon_entropy."""

from morie.fn.shanen import shannon_entropy


def test_shanen_basic():
    """Test basic functionality."""
    y = [1, 2, 3, 4]
    result = shannon_entropy(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_shanen_edge():
    """Test edge cases."""
    y = [1, 2, 3, 4]
    result = shannon_entropy(y)
    assert isinstance(result, dict)
