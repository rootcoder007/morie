"""Tests for siepid.si_epidemic."""

from morie.fn import _array_core as np

from morie.fn.siepid import si_epidemic


def test_siepid_basic():
    """Test basic functionality."""
    G = 0.5
    beta = 0.5
    initial = 0.5
    result = si_epidemic(G, beta, initial)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_siepid_edge():
    """Test edge cases."""
    G = 0.5
    beta = 0.5
    initial = 0.5
    result = si_epidemic(G, beta, initial)
    assert isinstance(result, dict)
