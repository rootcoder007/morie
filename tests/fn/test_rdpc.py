"""Tests for rdpc.renyi_dp."""

from morie.fn import _array_core as np

from morie.fn.rdpc import renyi_dp


def test_rdpc_basic():
    """Test basic functionality."""
    alpha = 5
    sigma = 0.5
    result = renyi_dp(alpha, sigma)
    assert isinstance(result, dict)
    assert "estimate" in result or "epsilon_rdp" in result


def test_rdpc_edge():
    """Test edge cases."""
    alpha = 5
    sigma = 0.5
    result = renyi_dp(alpha, sigma)
    assert isinstance(result, dict)
