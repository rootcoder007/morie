"""Tests for cominf.infomap."""

from morie.fn import _array_core as np

from morie.fn.cominf import infomap


def test_cominf_basic():
    """Test basic functionality."""
    G = np.eye(10)
    result = infomap(G)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_cominf_edge():
    """Test edge cases."""
    G = np.eye(10)
    result = infomap(G)
    assert isinstance(result, dict)
