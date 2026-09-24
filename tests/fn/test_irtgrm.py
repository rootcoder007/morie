"""Tests for irtgrm.graded_response."""

from morie.fn import _array_core as np

from morie.fn.irtgrm import graded_response


def test_irtgrm_basic():
    """Test basic functionality."""
    y = 0.5
    theta = 0.5
    a = 0.5
    b_k = 0.5
    result = graded_response(y, theta, a, b_k)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_irtgrm_edge():
    """Test edge cases."""
    y = 0.5
    theta = 0.5
    a = 0.5
    b_k = 0.5
    result = graded_response(y, theta, a, b_k)
    assert isinstance(result, dict)
