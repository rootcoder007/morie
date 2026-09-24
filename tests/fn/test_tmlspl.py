"""Tests for tmlspl.tmle_spillover."""

from morie.fn import _array_core as np

from morie.fn.tmlspl import tmle_spillover


def test_tmlspl_basic():
    """Test basic functionality."""
    y = 0.5
    D = 0.5
    X = 0.5
    network = 0.5
    result = tmle_spillover(y, D, X, network)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tmlspl_edge():
    """Test edge cases."""
    y = 0.5
    D = 0.5
    X = 0.5
    network = 0.5
    result = tmle_spillover(y, D, X, network)
    assert isinstance(result, dict)
