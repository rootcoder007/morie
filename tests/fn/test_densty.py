"""Tests for densty.density."""

from morie.fn import _array_core as np

from morie.fn.densty import density


def test_densty_basic():
    """Test basic functionality."""
    G = np.eye(10)
    result = density(G)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_densty_edge():
    """Test edge cases."""
    G = np.eye(10)
    result = density(G)
    assert isinstance(result, dict)
