"""Tests for grsnt.geron_sentiment_binary."""

from morie.fn import _array_core as np

from morie.fn.grsnt import geron_sentiment_binary


def test_grsnt_basic():
    """Test basic functionality."""
    token_ids = [0, 1]
    E = [[1.0], [2.0]]
    w = [1.0]
    result = geron_sentiment_binary(token_ids, E, w)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grsnt_edge():
    """Test edge cases."""
    token_ids = [0, 1]
    E = [[1.0], [2.0]]
    w = [1.0]
    result = geron_sentiment_binary(token_ids, E, w)
    assert isinstance(result, dict)
