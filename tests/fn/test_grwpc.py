"""Tests for grwpc.geron_wordpiece_tokenizer_score."""

from morie.fn import _array_core as np

from morie.fn.grwpc import geron_wordpiece_tokenizer_score


def test_grwpc_basic():
    """Test basic functionality."""
    counts = {'h': 15, 'u': 20, 'g': 4, 's': 5}
    pairs = {('h', 'u'): 10, ('g', 's'): 4}
    result = geron_wordpiece_tokenizer_score(counts, pairs)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grwpc_edge():
    """Test edge cases."""
    counts = {'h': 15, 'u': 20, 'g': 4, 's': 5}
    pairs = {('h', 'u'): 10, ('g', 's'): 4}
    result = geron_wordpiece_tokenizer_score(counts, pairs)
    assert isinstance(result, dict)
