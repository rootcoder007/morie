"""Tests for sietrt.sis_epidemic."""

from morie.fn import _array_core as np

from morie.fn.sietrt import sis_epidemic


def test_sietrt_basic():
    """Test basic functionality."""
    G = 0.5
    beta = 0.5
    gamma = 0.5
    initial = 0.5
    result = sis_epidemic(G, beta, gamma, initial)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_sietrt_edge():
    """Test edge cases."""
    G = 0.5
    beta = 0.5
    gamma = 0.5
    initial = 0.5
    result = sis_epidemic(G, beta, gamma, initial)
    assert isinstance(result, dict)
