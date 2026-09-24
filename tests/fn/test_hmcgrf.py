"""Tests for hmcgrf.geron_computational_graph."""

from morie.fn import _array_core as np

from morie.fn.hmcgrf import geron_computational_graph


def test_hmcgrf_basic():
    """Test basic functionality."""
    expr = 0.5
    result = geron_computational_graph(expr)
    assert isinstance(result, dict)
    assert "estimate" in result or "value" in result


def test_hmcgrf_edge():
    """Test edge cases."""
    expr = 0.5
    result = geron_computational_graph(expr)
    assert isinstance(result, dict)
