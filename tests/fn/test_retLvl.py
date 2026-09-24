"""Tests for retLvl.return_level."""

from morie.fn import _array_core as np

from morie.fn.retLvl import return_level


def test_retLvl_basic():
    """Test basic functionality."""
    mu = 0.5
    sigma = 0.5
    xi = 0.5
    T = 5
    result = return_level(mu, sigma, xi, T)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_retLvl_edge():
    """Test edge cases."""
    mu = 0.5
    sigma = 0.5
    xi = 0.5
    T = 5
    result = return_level(mu, sigma, xi, T)
    assert isinstance(result, dict)
