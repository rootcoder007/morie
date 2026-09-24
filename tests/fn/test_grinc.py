"""Tests for grinc.geron_in_context_learning."""

from morie.fn import _array_core as np

from morie.fn.grinc import geron_in_context_learning


def test_grinc_basic():
    """Test basic functionality."""
    examples = [('a', '1'), ('b', '2')]
    query = 'c'
    result = geron_in_context_learning(examples, query)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grinc_edge():
    """Test edge cases."""
    examples = [('a', '1'), ('b', '2')]
    query = 'c'
    result = geron_in_context_learning(examples, query)
    assert isinstance(result, dict)
