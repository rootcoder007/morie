"""Tests for netcls.closeness_centrality."""

from morie.fn import _array_core as np

from morie.fn.netcls import closeness_centrality


def test_netcls_basic():
    """Test basic functionality."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = closeness_centrality(A)
    assert isinstance(result, dict)
    assert "closeness" in result


def test_netcls_edge():
    """Test edge cases."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = closeness_centrality(A)
    assert isinstance(result, dict)
