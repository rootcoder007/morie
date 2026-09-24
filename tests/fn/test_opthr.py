"""Tests for opthr.optimal_huber_k."""

from morie.fn import _array_core as np

from morie.fn.opthr import optimal_huber_k


def test_opthr_basic():
    """Test basic functionality."""
    result = optimal_huber_k()
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_opthr_edge():
    """Test edge cases."""
    result = optimal_huber_k()
    assert isinstance(result, dict)
