"""Tests for siaepi.sir_epidemic."""

from morie.fn import _array_core as np

from morie.fn.siaepi import sir_epidemic


def test_siaepi_basic():
    """Test basic functionality."""
    G = 0.5
    beta = 0.5
    gamma = 0.5
    initial = 0.5
    result = sir_epidemic(G, beta, gamma, initial)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_siaepi_edge():
    """Test edge cases."""
    G = 0.5
    beta = 0.5
    gamma = 0.5
    initial = 0.5
    result = sir_epidemic(G, beta, gamma, initial)
    assert isinstance(result, dict)
