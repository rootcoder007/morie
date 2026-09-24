"""Tests for netcmp.network_comparison."""

from morie.fn import _array_core as np

from morie.fn.netcmp import network_comparison


def test_netcmp_basic():
    """Test basic functionality."""
    G1 = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    G2 = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = network_comparison(G1, G2)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_netcmp_edge():
    """Test edge cases."""
    G1 = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    G2 = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = network_comparison(G1, G2)
    assert isinstance(result, dict)
