"""Tests for sbreli.spearman_brown."""

from morie.fn.sbreli import spearman_brown


def test_sbreli_basic():
    """Test basic functionality."""
    r = 10
    k = 5
    result = spearman_brown(r, k)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_sbreli_edge():
    """Test edge cases."""
    r = 10
    k = 5
    result = spearman_brown(r, k)
    assert isinstance(result, dict)
