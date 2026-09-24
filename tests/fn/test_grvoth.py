"""Tests for grvoth.geron_hard_voting."""

from morie.fn import _array_core as np

from morie.fn.grvoth import geron_hard_voting


def test_grvoth_basic():
    """Test basic functionality."""
    predictions = [[0], [1]]
    result = geron_hard_voting(predictions)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grvoth_edge():
    """Test edge cases."""
    predictions = [[0], [1]]
    result = geron_hard_voting(predictions)
    assert isinstance(result, dict)
