"""Tests for linkPr.link_prediction."""

from morie.fn import _array_core as np

from morie.fn.linkpr import link_prediction


def test_linkpr_basic():
    """Test basic functionality."""
    G = 0.5
    u = 0.5
    v = 0.5
    result = link_prediction(G, u, v)
    assert isinstance(result, dict)
    assert "estimate" in result or "common_neighbours" in result


def test_linkpr_edge():
    """Test edge cases."""
    G = 0.5
    u = 0.5
    v = 0.5
    result = link_prediction(G, u, v)
    assert isinstance(result, dict)
